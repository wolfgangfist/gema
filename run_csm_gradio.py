import os
os.environ["NO_TORCH_COMPILE"] = "1"

try:
    import triton
except ImportError:
    print("Warning: Triton not available, using CPU-only mode")

import gradio as gr
import numpy as np
import torch
import torchaudio
from generator import Segment, load_csm_1b
from huggingface_hub import hf_hub_download

SPACE_INTRO_TEXT = """\
# Sesame CSM 1B Demo

Generate conversations using CSM 1B (Conversational Speech Model). 
Each line in the conversation will alternate between Speaker A and B.
"""

DEFAULT_CONVERSATION = """\
Hey how are you doing.
Pretty good, pretty good.
I'm great, so happy to be speaking to you.
Me too, this is some cool stuff huh?
"""

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam"
        ),
        "audio": "prompts/conversational_a.wav",
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game"
        ),
        "audio": "prompts/conversational_b.wav",
    }
}

device = "cpu"  # Force CPU for compatibility
model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, device)

def prepare_prompt(text: str, speaker: int, audio_path: str) -> Segment:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    if sample_rate != generator.sample_rate:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate
        )
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def infer(
    text_prompt_speaker_a,
    text_prompt_speaker_b,
    audio_prompt_speaker_a,
    audio_prompt_speaker_b,
    conversation_input,
) -> tuple[int, np.ndarray]:
    try:
        # Estimate token limit
        if len(conversation_input.strip() + text_prompt_speaker_a.strip() + text_prompt_speaker_b.strip()) >= 2000:
            raise gr.Error("Prompts and conversation too long.")

        # Prepare prompts
        audio_prompt_a = prepare_prompt(text_prompt_speaker_a, 0, audio_prompt_speaker_a)
        audio_prompt_b = prepare_prompt(text_prompt_speaker_b, 1, audio_prompt_speaker_b)
        prompt_segments = [audio_prompt_a, audio_prompt_b]
        
        # Generate conversation
        generated_segments = []
        conversation_lines = [line.strip() for line in conversation_input.strip().split("\n") if line.strip()]
        
        for i, line in enumerate(conversation_lines):
            # Alternating speakers A and B
            speaker_id = i % 2
            
            audio_tensor = generator.generate(
                text=line,
                speaker=speaker_id,
                context=prompt_segments + generated_segments,
                max_audio_length_ms=30_000,
            )
            generated_segments.append(Segment(text=line, speaker=speaker_id, audio=audio_tensor))

        # Concatenate all generations
        audio_tensors = [segment.audio for segment in generated_segments]
        audio_tensor = torch.cat(audio_tensors, dim=0)
        
        # Convert to numpy array
        audio_array = (audio_tensor * 32768).to(torch.int16).cpu().numpy()
        
        return generator.sample_rate, audio_array

    except Exception as e:
        raise gr.Error(f"Error generating audio: {e}")

def create_speaker_prompt_ui(speaker_name: str):
    speaker_dropdown = gr.Dropdown(
        choices=list(SPEAKER_PROMPTS.keys()),
        label="Select a predefined speaker",
        value=speaker_name
    )
    with gr.Accordion("Or add your own voice prompt", open=False):
        text_prompt_speaker = gr.Textbox(
            label="Speaker prompt",
            lines=4,
            value=SPEAKER_PROMPTS[speaker_name]["text"]
        )
        audio_prompt_speaker = gr.Audio(
            label="Speaker prompt",
            type="filepath",
            value=SPEAKER_PROMPTS[speaker_name]["audio"]
        )

    return speaker_dropdown, text_prompt_speaker, audio_prompt_speaker

def update_prompt(speaker):
    if speaker in SPEAKER_PROMPTS:
        return (
            SPEAKER_PROMPTS[speaker]["text"],
            SPEAKER_PROMPTS[speaker]["audio"]
        )
    return None, None

with gr.Blocks() as app:
    gr.Markdown(SPACE_INTRO_TEXT)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Speaker A")
            speaker_a_dropdown, text_prompt_speaker_a, audio_prompt_speaker_a = create_speaker_prompt_ui(
                "conversational_a"
            )

        with gr.Column():
            gr.Markdown("### Speaker B")
            speaker_b_dropdown, text_prompt_speaker_b, audio_prompt_speaker_b = create_speaker_prompt_ui(
                "conversational_b"
            )

    # Update prompts when dropdown changes
    speaker_a_dropdown.change(
        fn=update_prompt,
        inputs=[speaker_a_dropdown],
        outputs=[text_prompt_speaker_a, audio_prompt_speaker_a]
    )
    speaker_b_dropdown.change(
        fn=update_prompt,
        inputs=[speaker_b_dropdown],
        outputs=[text_prompt_speaker_b, audio_prompt_speaker_b]
    )

    gr.Markdown("## Conversation")
    conversation_input = gr.TextArea(
        label="Enter conversation (alternating lines between speakers)",
        lines=10,
        value=DEFAULT_CONVERSATION
    )
    
    generate_btn = gr.Button("Generate conversation", variant="primary")
    audio_output = gr.Audio(label="Generated conversation")

    generate_btn.click(
        fn=infer,
        inputs=[
            text_prompt_speaker_a,
            text_prompt_speaker_b,
            audio_prompt_speaker_a,
            audio_prompt_speaker_b,
            conversation_input,
        ],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    app.launch() 