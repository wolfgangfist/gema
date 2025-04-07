from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Generator
import time
import os

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

os.environ["NO_TORCH_COMPILE"] = "1"


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

        self.sample_rate = mimi.sample_rate
        self.device = device
        
        # Cache for tokenized prompts
        self._prompt_cache = {}

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (text, speaker)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33, device=self.device).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33, device=self.device).bool()
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame)
        frame_masks.append(text_frame_mask)

        result = (torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0))
        self._prompt_cache[cache_key] = result
        return result

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # Keep audio on device
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1, device=self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33, device=self.device).long()
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33, device=self.device).bool()
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
    ) -> torch.Tensor:
        """Generate audio from text."""
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        # Pre-allocate tensors for generation
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long()  # Already on device from tokenization
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool()  # Already on device

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0).long()

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        # Pre-allocate tensors for frame generation
        zeros_1_1 = torch.zeros(1, 1, device=self.device).long()
        zeros_1_1_bool = torch.zeros(1, 1, device=self.device).bool()

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            # Use pre-allocated tensors and avoid unnecessary concatenations
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample, dtype=torch.bool), zeros_1_1_bool], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        # Stack all samples at once
        if samples:
            stacked_samples = torch.stack(samples).permute(1, 2, 0)
            audio = self._audio_tokenizer.decode(stacked_samples).squeeze(0).squeeze(0)
            return audio
        else:
            return torch.zeros(1, device=self.device)  # Return empty audio if no samples


def load_csm_1b(device: str = "cuda") -> Generator:
    print("Loading CSM-1B model...")
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    if os.environ.get("NO_TORCH_COMPILE", "0") == "1":
        print("Skipping torch.compile based on NO_TORCH_COMPILE environment variable.")
    else:
        print("Compiling model with basic mode...")
        compile_start = time.time()
        try:
            # Use basic mode instead of max-autotune to avoid Triton issues
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print(f"Model compiled successfully in {time.time() - compile_start:.2f} seconds.")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}. Continuing without compilation.")

    generator = Generator(model)
    return generator