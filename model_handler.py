import torch
import torchaudio
import time
from huggingface_hub import hf_hub_download
from typing import Generator as PyGenerator, Optional, List, Union

try:
    from generator import load_csm_1b, Segment, Generator
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import generator components: {e}")
    MODEL_AVAILABLE = False

class VoiceGenerator:
    def __init__(self) -> None:
        """Initialize device and basic attributes."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    print("Flash Attention enabled.")
                except Exception as e:
                    print(f"Warning: Could not enable Flash Attention: {e}")
        
        self.generator: Optional[Generator] = None
        self.prompt_segments: List[Segment] = []
        self.sample_rate: int = 24000

    def initialize(self) -> None:
        """Load model, prepare prompts, and warm up the model."""
        if not MODEL_AVAILABLE:
            print("Skipping model initialization due to import failure.")
            return
        
        try:
            # Load the model on the specified device
            self.generator = load_csm_1b(self.device)
            self.sample_rate = self.generator.sample_rate
            print(f"Model loaded on {self.device}. Sample rate: {self.sample_rate}")

            # Prepare prompt segments
            prompt_a_path = hf_hub_download("sesame/csm-1b", "prompts/conversational_a.wav")
            prompt_a_text = "like revising for an exam I'd have to try and like keep up the momentum"
            prompt_a = self.prepare_prompt(prompt_a_text, 0, prompt_a_path)
            
            prompt_b_path = hf_hub_download("sesame/csm-1b", "prompts/conversational_b.wav")
            prompt_b_text = "like a super Mario level. Like it's very like high detail."
            prompt_b = self.prepare_prompt(prompt_b_text, 1, prompt_b_path)
            
            self.prompt_segments = [prompt_a, prompt_b]
            print(f"Prepared {len(self.prompt_segments)} prompts.")

            # Warm up the model
            self.warm_up()
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            self.generator = None

    def prepare_prompt(self, text: str, speaker: int, audio_path: str) -> Segment:
        """Prepare a prompt segment from text and audio file."""
        audio_tensor = self.load_prompt_audio(audio_path)
        return Segment(text=text, speaker=speaker, audio=audio_tensor)

    def load_prompt_audio(self, audio_path: str) -> torch.Tensor:
        """Load and resample audio to the model's sample rate."""
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.squeeze(0)
        if sample_rate != self.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=self.sample_rate
            )
        return audio_tensor

    def warm_up(self) -> None:
        """Warm up the model to reduce time to first chunk."""
        if self.generator is None:
            return
        
        print("Warming up model...")
        dummy_text = "This is a warm-up text to prepare the model for fast responses."
        speaker_id = 0
        context = [self.prompt_segments[speaker_id]] if speaker_id < len(self.prompt_segments) else self.prompt_segments
        
        try:
            _ = self.generator.generate(
                text=dummy_text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=5000,
                temperature=0.9,
                topk=50
            )
            print("Model warm-up complete.")
        except Exception as e:
            print(f"Warning: Model warm-up failed: {e}")

    def generate_pcm_stream(
        self, 
        text: str, 
        speaker_id: int, 
        target_sample_rate: int,
        max_audio_length_ms: int = 30_000,
        temperature: float = 0.9,
        topk: int = 50,
        stream_chunk_frames: int = 10,
    ) -> PyGenerator[bytes, None, None]:
        """Generate a stream of PCM bytes at the target sample rate."""
        if self.generator is None:
            raise RuntimeError("Generator not initialized.")

        print(f"\n--- Starting PCM Stream Generation ---")
        print(f"Input text: '{text}'")
        print(f"Speaker ID: {speaker_id}")
        stream_start_time = time.monotonic()
        chunk_count = 0
        first_chunk_yielded = False

        try:
            context = [self.prompt_segments[speaker_id]] if speaker_id < len(self.prompt_segments) else self.prompt_segments
            print(f"Using context with {len(context)} prompt segments")
            
            # Generate complete audio first
            audio_data = self.generator.generate(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                topk=topk
            )

            # Split into chunks and stream
            chunk_size = int(self.sample_rate * 0.08 * stream_chunk_frames)  # 80ms per frame
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                if not isinstance(chunk, torch.Tensor) or chunk.numel() == 0:
                    continue

                try:
                    if self.sample_rate != target_sample_rate:
                        resampled_chunk = torchaudio.functional.resample(
                            chunk.cpu(), 
                            orig_freq=self.sample_rate, 
                            new_freq=target_sample_rate
                        )
                    else:
                        resampled_chunk = chunk.cpu()

                    pcm_bytes = (resampled_chunk * 32767).short().numpy().tobytes()
                    if not pcm_bytes:
                        continue

                    yield pcm_bytes
                    chunk_count += 1

                    if not first_chunk_yielded:
                        ttfc_ms = (time.monotonic() - stream_start_time) * 1000
                        print(f"  TTFC: {ttfc_ms:.2f} ms | First PCM chunk ({len(pcm_bytes)} bytes)")
                        first_chunk_yielded = True

                except Exception as e:
                    print(f"  ERROR processing chunk {chunk_count}: {e}")
                    continue

        except Exception as e:
            print(f"!! ERROR during PCM stream generation: {e}")
            raise
        finally:
            total_duration = time.monotonic() - stream_start_time
            if chunk_count > 0:
                print(f"PCM Stream finished in {total_duration:.3f}s. Generated {chunk_count} chunks @ {target_sample_rate}Hz.")
            else:
                print(f"PCM Stream finished after {total_duration:.3f}s with no chunks generated.")

    def generate_pcm_full(
        self,
        text: str,
        speaker_id: int,
        target_sample_rate: int,
        max_audio_length_ms: int = 30_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> bytes:
        """Generate complete PCM audio as a single response."""
        if self.generator is None:
            raise RuntimeError("Generator not initialized.")

        print(f"\n--- Starting Full PCM Generation ---")
        print(f"Input text: '{text}'")
        print(f"Speaker ID: {speaker_id}")
        generation_start_time = time.monotonic()

        try:
            context = [self.prompt_segments[speaker_id]] if speaker_id < len(self.prompt_segments) else self.prompt_segments
            print(f"Using context with {len(context)} prompt segments")
            
            print("Generating audio...")
            audio_data = self.generator.generate(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                topk=topk
            )

            print(f"Audio tensor generated - shape: {audio_data.shape}")

            # Resample if needed
            if self.sample_rate != target_sample_rate:
                print(f"Resampling from {self.sample_rate}Hz to {target_sample_rate}Hz")
                audio_data = torchaudio.functional.resample(
                    audio_data.cpu(),
                    orig_freq=self.sample_rate,
                    new_freq=target_sample_rate
                )
            else:
                audio_data = audio_data.cpu()

            # Convert to PCM bytes
            pcm_bytes = (audio_data * 32767).short().numpy().tobytes()
            
            generation_time = time.monotonic() - generation_start_time
            print(f"Full PCM generation completed in {generation_time:.3f}s @ {target_sample_rate}Hz")
            print(f"Generated {len(pcm_bytes)} bytes of audio data")
            
            return pcm_bytes

        except Exception as e:
            print(f"!! ERROR during full PCM generation: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise 