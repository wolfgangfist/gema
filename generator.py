from dataclasses import dataclass
import os
from typing import List, Tuple, Generator as PyGenerator, Optional, Callable
import time
import queue
import threading
import platform
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

# Advanced CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()

# Windows-specific optimizations (keep your existing ones too)
if os.name == 'nt':
    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.conv_1x1_as_mm = True
    
# Thread optimization
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(self, model: Model):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()
        device = next(model.parameters()).device

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

        self._stream_buffer_size = 10
        self.max_seq_len = 2048

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        total_len = text_tokens.size(0) + audio_tokens.size(0)

        if total_len > self.max_seq_len:
            overflow = total_len - self.max_seq_len

            if text_tokens.size(0) > overflow:
                text_tokens = text_tokens[overflow:]
                text_masks = text_masks[overflow:]
            else:
                audio_overflow = overflow - text_tokens.size(0)
                text_tokens = text_tokens[0:0]
                text_masks = text_masks[0:0]
                audio_tokens = audio_tokens[audio_overflow:]
                audio_masks = audio_masks[audio_overflow:]

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    
    def _decode_frames(self, frames):
        """Decode a batch of frames into audio with optimized memory handling"""
        if not frames:
            return torch.tensor([])
        
        audio = self._audio_tokenizer.decode(torch.stack(frames).permute(1, 2, 0)).squeeze(0).squeeze(0)

        return audio 

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> PyGenerator[torch.Tensor, None, None]:
        import threading
        import queue

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        if context:
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        if prompt_tokens.size(0) > self.max_seq_len:
            prompt_tokens = prompt_tokens[-self.max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-self.max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        frame_queue = queue.Queue()
        audio_queue = queue.Queue()
        stop_event = threading.Event()
        decode_stride = 60
        batch_rollout = 8  # how many frames to rollout at once
        i = 0

        @torch.inference_mode()
        def decode_worker():
            frame_buffer = []
            while not stop_event.is_set() or not frame_queue.empty():
                try:
                    sample = frame_queue.get(timeout=0.1)
                    if sample is None:
                        break
                    frame_buffer.append(sample)
                    if len(frame_buffer) >= decode_stride:
                        audio_chunk = self._audio_tokenizer.decode(
                            torch.stack(frame_buffer).permute(1, 2, 0)
                        ).squeeze(0).squeeze(0)
                        frame_buffer.clear()
                        audio_queue.put(audio_chunk.detach().cpu())
                except queue.Empty:
                    continue

            if frame_buffer:
                audio_chunk = self._audio_tokenizer.decode(
                    torch.stack(frame_buffer).permute(1, 2, 0)
                ).squeeze(0).squeeze(0)
                audio_queue.put(audio_chunk.detach().cpu())

        decode_thread = threading.Thread(target=decode_worker, daemon=True)
        decode_thread.start()

        print("Starting threaded streaming generation...")
        generation_start = time.time()

        with self._audio_tokenizer.streaming(1):
            zeros_1_1 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            zeros_mask_1_1 = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

            while i < max_generation_len:
                actual_rollout = min(batch_rollout, max_generation_len - i)
                batch_samples = []

                for _ in range(actual_rollout):
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16), torch.no_grad():
                        sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    if torch.all(sample == 0):
                        break
                    batch_samples.append(sample)

                    # Update in-place
                    curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
                    curr_tokens_mask = torch.cat(
                        [torch.ones_like(sample, dtype=torch.bool), zeros_mask_1_1], dim=1
                    ).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1

                if not batch_samples:
                    break

                for s in batch_samples:
                    frame_queue.put(s)
                i += len(batch_samples)

        stop_event.set()
        frame_queue.put(None)
        decode_thread.join(timeout=5.0)

        while not audio_queue.empty():
            chunk = audio_queue.get()
            if on_chunk_generated:
                on_chunk_generated(chunk)
            yield chunk

        total_time = time.time() - generation_start
        audio_seconds = i * 0.08
        rtf = audio_seconds / total_time if total_time > 0 else float('inf')
        inv_rtf = total_time / audio_seconds if audio_seconds > 0 else float('inf')

        print("Threaded audio generation complete")
        print(f"Model + decode time: {total_time:.2f}s for {audio_seconds:.2f}s of audio")
        print(f"Real-time factor: {rtf:.2f}x (RTF)")
        print(f"Compute time per audio second: {inv_rtf:.3f}s (target < 1.0)")


    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        stream: bool = False,
    ) -> torch.Tensor:
        if stream:
            audio_chunks = []
            for chunk in self.generate_stream(text, speaker, context, max_audio_length_ms, temperature, topk):
                audio_chunks.append(chunk)
            if not audio_chunks:
                return torch.tensor([])
            return torch.cat(audio_chunks)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        max_seq_len = 2048
        if prompt_tokens.size(0) > max_seq_len:
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        samples = []
        with self._audio_tokenizer.streaming(1):
            for _ in range(max_generation_len):
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    break
                samples.append(sample)

                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            return torch.tensor([])

        return self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)



class AudioStreamWriter:
    """
    Helper class for writing streaming audio to a file.
    """
    def __init__(self, filename, sample_rate):
        self.filename = filename
        self.sample_rate = sample_rate
        self.audio_chunks = []
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.running = True
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()
        
    def _writer_worker(self):
        """Background thread that handles audio chunk processing"""
        buffer_chunks = []
        last_flush_time = time.time()
        
        while self.running or not self.queue.empty():
            try:
                # Get chunk with timeout to allow for regular checks
                chunk = self.queue.get(timeout=0.2)
                buffer_chunks.append(chunk)
                
                # Periodically flush the buffer to the main list
                current_time = time.time()
                if len(buffer_chunks) >= 10 or (current_time - last_flush_time > 2.0 and buffer_chunks):
                    with self.lock:
                        self.audio_chunks.extend(buffer_chunks)
                    buffer_chunks = []
                    last_flush_time = current_time
                    
            except queue.Empty:
                # If queue is empty but we have pending chunks, add them
                if buffer_chunks:
                    with self.lock:
                        self.audio_chunks.extend(buffer_chunks)
                    buffer_chunks = []
                    last_flush_time = time.time()
        
        # Final flush of any remaining chunks
        if buffer_chunks:
            with self.lock:
                self.audio_chunks.extend(buffer_chunks)
        
    def add_chunk(self, chunk):
        """Add an audio chunk to the buffer queue without blocking"""
        try:
            self.queue.put(chunk, timeout=0.1)
        except queue.Full:
            # If queue is full, add directly to avoid losing data
            with self.lock:
                self.audio_chunks.append(chunk)
    
    def write_file(self):
        """Write all collected audio chunks to file and clean up"""
        # Signal the background thread to stop
        self.running = False
        # Wait for the thread to finish with a timeout
        self.writer_thread.join(timeout=3.0)
        
        with self.lock:
            if not self.audio_chunks:
                return
                
            # Concatenate all chunks
            audio = torch.cat(self.audio_chunks)
            # Save to file
            torchaudio.save(self.filename, audio.unsqueeze(0).cpu(), self.sample_rate)

from safetensors.torch import load_file
import os
import torch
from models import Model, ModelArgs
from generator import Generator

def load_csm_1b_local(model_path: str, device: str = "cuda"):
    """
    Load the CSM-1B model from a local .safetensors checkpoint with extreme optimizations.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    print(f"Loading CSM-1B model from local checkpoint '{model_path}' with extreme optimizations...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )

    model = Model(config)
    
    safetensor_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(safetensor_path, device=device)
    model.load_state_dict(state_dict, strict=True)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    model = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
        backend='cudagraphs'
    )
    
    model.to(device=device, dtype=dtype)
    
    if torch.cuda.is_available():
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                dummy_input = torch.zeros(1, 10, 33, device=device, dtype=torch.long)
                dummy_mask = torch.ones(1, 10, 33, device=device, dtype=torch.bool)
                dummy_pos = torch.arange(10, device=device).unsqueeze(0)
                _ = model(dummy_input, dummy_mask, dummy_pos)
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warm-up failed, but continuing anyway: {e}")

    print("Model compilation complete. Creating generator...")
    
    generator = Generator(model)
    
    generator._stream_buffer_size = 30 
    
    generator._tokenization_cache = {}
    
    original_tokenize_text = generator._tokenize_text_segment
    
    def patched_tokenize_text_segment(text, speaker):
        cache_key = f"{speaker}:{text}"
        if cache_key in generator._tokenization_cache:
            return generator._tokenization_cache[cache_key]
        
        result = original_tokenize_text(text, speaker)
        generator._tokenization_cache[cache_key] = result
        return result
    
    generator._tokenize_text_segment = patched_tokenize_text_segment
    
    return generator


def load_csm_1b(device: str = "cuda") -> Generator:
    """
    Load the CSM-1B model with extreme optimizations for real-time performance.
    """
    # Enable all CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    print("Loading CSM-1B model with extreme optimizations for real-time performance...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    model = Model.from_pretrained("sesame/csm-1b")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    model = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
        backend='cudagraphs'
    )
    
    model.to(device=device, dtype=dtype)
    
    
    if torch.cuda.is_available():
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                dummy_input = torch.zeros(1, 10, 33, device=device, dtype=torch.long)
                dummy_mask = torch.ones(1, 10, 33, device=device, dtype=torch.bool)
                dummy_pos = torch.arange(10, device=device).unsqueeze(0)
                _ = model(dummy_input, dummy_mask, dummy_pos)
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warm-up failed, but continuing anyway: {e}")
    
    print("Model compilation complete. Creating generator...")
    
    generator = Generator(model)
    
    generator._stream_buffer_size = 30
    
    
    generator._tokenization_cache = {}
    
    original_tokenize_text = generator._tokenize_text_segment
    
    def patched_tokenize_text_segment(text, speaker):
        cache_key = f"{speaker}:{text}"
        if cache_key in generator._tokenization_cache:
            return generator._tokenization_cache[cache_key]
        
        result = original_tokenize_text(text, speaker)
        generator._tokenization_cache[cache_key] = result
        return result
    
    generator._tokenize_text_segment = patched_tokenize_text_segment
    
    return generator


def generate_streaming_audio(
    generator: Generator,
    text: str,
    speaker: int,
    context: List[Segment],
    output_file: str,
    max_audio_length_ms: float = 90_000,
    temperature: float = 0.8,
    topk: int = 50,
    play_audio: bool = False,
):
    import os
    import time
    import threading
    import collections
    import tempfile
    import subprocess
    import torchaudio
    import platform
    from threading import Lock

    os.environ['OMP_NUM_THREADS'] = '1'

    writer_chunks = []
    audio_ring_buffer = collections.deque()
    stop_event = threading.Event()
    total_audio_samples = 0

    # -- Playback --
    if play_audio:
        try:
            import sounddevice as sd
            import numpy as np

            def playback_worker():
                stream = sd.OutputStream(samplerate=generator.sample_rate, channels=1, blocksize=2048)
                stream.start()
                while not stop_event.is_set() or audio_ring_buffer:
                    if audio_ring_buffer:
                        chunk = audio_ring_buffer.popleft()
                        stream.write(chunk.detach().cpu().numpy())
                    else:
                        time.sleep(0.01)
                stream.write(np.zeros(int(generator.sample_rate * 0.5), dtype=np.float32))
                stream.stop()
                stream.close()

            player_thread = threading.Thread(target=playback_worker, daemon=True)
            player_thread.start()
        except ImportError:
            print("Install 'sounddevice' for playback.")
            play_audio = False

    # -- Chunk writer --
    tmp_dir = tempfile.mkdtemp()
    chunk_paths = []
    chunk_idx = 0
    writer_lock = Lock()

    def write_chunk(chunk: torch.Tensor):
        nonlocal chunk_idx, total_audio_samples
        with writer_lock:
            chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_idx}.wav")
            torchaudio.save(chunk_path, chunk.unsqueeze(0).detach().cpu(), generator.sample_rate)
            chunk_paths.append(chunk_path)
            total_audio_samples += chunk.size(0)
            chunk_idx += 1

    # -- On chunk callback --
    def on_chunk_generated(chunk):
        write_chunk(chunk)
        if play_audio:
            audio_ring_buffer.append(chunk)

    # -- High priority --
    try:
        import psutil
        proc = psutil.Process()
        if platform.system() == "Windows":
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            proc.nice(0)
    except Exception:
        pass

    print("Generating audio...")
    gen_start = time.time()

    # -- Run model generation only --
    for _ in generator.generate_stream(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk,
        on_chunk_generated=on_chunk_generated,
    ):
        pass

    gen_end = time.time()
    stop_event.set()

    if play_audio:
        player_thread.join(timeout=3.0)

    # -- Combine audio chunks using ffmpeg or fallback --
    output_path_list = os.path.join(tmp_dir, "file_list.txt")
    with open(output_path_list, "w") as f:
        for path in chunk_paths:
            f.write(f"file '{path}'\n")

    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", output_path_list,
            "-c", "copy", output_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        full_audio = torch.cat(writer_chunks)
        torchaudio.save(output_file, full_audio.unsqueeze(0).cpu(), generator.sample_rate)

    for path in chunk_paths:
        try:
            os.remove(path)
        except:
            pass
    try:
        os.remove(output_path_list)
        os.rmdir(tmp_dir)
    except:
        pass

    audio_seconds = total_audio_samples / generator.sample_rate
    model_time = gen_end - gen_start
    rtf = audio_seconds / model_time if model_time > 0 else float('inf')
    inv_rtf = model_time / audio_seconds if audio_seconds > 0 else float('inf')

    print("Audio generation done")
    print(f"Model generation time: {model_time:.2f}s for {audio_seconds:.2f}s of audio")
    print(f"Real-time factor: {rtf:.2f}x (RTF)")
    print(f"Compute time per audio second: {inv_rtf:.3f}s (target < 1.0)")

