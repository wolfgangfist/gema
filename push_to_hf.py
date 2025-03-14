from models import Model, ModelArgs
from generator import Generator

from huggingface_hub import hf_hub_download

import torch

def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    generator = Generator(model)
    return generator


if __name__ == "__main__":
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    model = Model(model_args)
    
    # equip with weights
    filepath = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    state_dict = torch.load(filepath, map_location="cpu")
    model.load_state_dict(state_dict)

    # push to the hub
    model.push_to_hub("nielsr/csm-1b")