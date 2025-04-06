from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
import time

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


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
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

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
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        return_timers: bool = False,
        disable_watermark: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Generates audio based on text, speaker, and context.

        Args:
            ... (other args) ...
            return_timers: If True, returns a dictionary with timing information.
            disable_watermark: If True, skips the watermarking step.

        Returns:
            Generated audio tensor, or (audio tensor, timers dict) if return_timers is True.
        """
        self._model.reset_caches()

        timers: Dict[str, Union[float, List[float]]] = {}
        start_total = time.time() if return_timers else None

        # --- Tokenization ---
        start_tokenize = time.time() if return_timers else None
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
        if return_timers: timers["tokenization"] = time.time() - start_tokenize

        start_generation = time.time() if return_timers else None
        samples = []
        frame_times = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = getattr(self._model.backbone, 'max_seq_len', 2048)
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Input context length ({curr_tokens.size(1)}) exceeds maximum "
                f"allowed ({max_context_len} = {max_seq_len} - {max_generation_len}). "
                f"Reduce context or increase max_audio_length_ms."
            )

        for i in range(max_generation_len):
            frame_start = time.time() if return_timers else None
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if return_timers: frame_times.append(time.time() - frame_start)
            if torch.all(sample == 0):
                if return_timers: print(f"EOS token detected at frame {i+1}.")
                break

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
            if curr_pos[0, -1] >= max_seq_len:
                print(f"Warning: Reached maximum sequence length ({max_seq_len}) during generation.")
                break

        if return_timers:
            timers["generation"] = time.time() - start_generation
            timers["frame_times"] = frame_times
            timers["num_frames"] = len(samples)

        # --- Audio Decoding ---
        if not samples:
            print("Warning: No audio samples generated.")
            audio = torch.empty((0,), dtype=torch.float32, device=self.device)
            if return_timers: timers["audio_decoding"] = 0.0
        else:
            start_decode = time.time() if return_timers else None
            stacked_samples = torch.stack(samples).permute(1, 2, 0)
            audio = self._audio_tokenizer.decode(stacked_samples).squeeze(0).squeeze(0)
            if return_timers: timers["audio_decoding"] = time.time() - start_decode

        # --- Watermarking (Optional) ---
        if not disable_watermark and self._watermarker is not None:
            start_watermark = time.time() if return_timers else None
            try:
                audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
                if wm_sample_rate != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
                if return_timers: timers["watermarking"] = time.time() - start_watermark
            except Exception as e:
                print(f"Warning: Watermarking failed - {e}")
                if return_timers: timers["watermarking"] = -1.0

        if return_timers:
            timers["total"] = time.time() - start_total
            return audio.detach(), timers
        else:
            return audio.detach()


def load_csm_1b(device: str = "cuda") -> Generator:
    print("Loading model...")
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    print("Compiling model (mode='max-autotune', fullgraph=True)...")
    compile_start = time.time()
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True, cudagraphs=True)
        print(f"Model compiled successfully in {time.time() - compile_start:.2f} seconds.")
    except Exception as e:
        print(f"Model compilation failed: {e}. Performance may be suboptimal.")

    generator = Generator(model)
    return generator