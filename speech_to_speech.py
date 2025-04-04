import os
import torch
import torchaudio
import numpy as np
from fastrtc import Stream, ReplyOnPause
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import gradio as gr

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

class CSMSpeechProcessor:
    def __init__(self, device="cuda"):
        # Initialize the CSM model
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.generator = load_csm_1b(device=self.device)
        
        # Download voice prompts
        self.prompt_a_path = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_a.wav"
        )
        self.prompt_b_path = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_b.wav"
        )
        
        # Load prompts for voice identity
        self.prompt_a = self._load_prompt(
            self.prompt_a_path,
            "like revising for an exam I'd have to try and like keep up the momentum because I'd start really early",
            0  # Speaker ID
        )
        self.prompt_b = self._load_prompt(
            self.prompt_b_path,
            "like a super Mario level. Like it's very like high detail. And like, once you get into the park",
            1  # Speaker ID
        )
        
        # Initialize conversation history
        self.context = [self.prompt_a, self.prompt_b]
        self.history = []
        
    def _load_prompt(self, audio_path, text, speaker_id):
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(
            audio.squeeze(0), orig_freq=sr, new_freq=self.generator.sample_rate
        )
        return Segment(text=text, speaker=speaker_id, audio=audio)
    
    def _convert_audio_format(self, fastrtc_audio):
        sample_rate, audio_array = fastrtc_audio
        # Convert to mono if needed
        if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Convert to torch tensor and resample if needed
        audio_tensor = torch.from_numpy(audio_array).float()
        if sample_rate != self.generator.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=self.generator.sample_rate
            )
        
        return audio_tensor
    
    def process_speech(self, audio):
        """Process incoming speech and generate a response"""
        # Convert FastRTC audio format to CSM format
        audio_tensor = self._convert_audio_format(audio)
        
        # Add to conversation history
        input_segment = Segment(text="", speaker=0, audio=audio_tensor)
        self.context.append(input_segment)
        
        # Generate response with CSM
        response_audio = self.generator.generate(
            text="",  # Empty for speech-to-speech mode
            speaker=1,  # Use the second speaker voice
            context=self.context,
            max_audio_length_ms=10_000,
            temperature=0.9,  # Control variability
        )
        
        # Add response to context
        response_segment = Segment(text="", speaker=1, audio=response_audio)
        self.context.append(response_segment)
        
        # Manage context length (keep most recent interactions)
        if len(self.context) > 6:  # Keep prompts + recent 4 exchanges
            self.context = self.context[:2] + self.context[-4:]
        
        # Return audio in FastRTC format
        return self.generator.sample_rate, response_audio.cpu().numpy()
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.context = [self.prompt_a, self.prompt_b]
        return "Conversation reset"

def main():
    # Initialize processor
    processor = CSMSpeechProcessor()

    # Create FastRTC stream
    stream = Stream(
        handler=ReplyOnPause(processor.process_speech),
        modality="audio",
        mode="send-receive",
    )

    # Create Gradio interface with additional controls
    with gr.Blocks() as demo:
        gr.Markdown("# CSM Speech-to-Speech Conversational AI")
        gr.Markdown("Speak naturally and the AI will respond when you pause")
        
        # Mount FastRTC stream UI
        stream.ui.render()
        
        with gr.Row():
            reset_btn = gr.Button("Reset Conversation")
        
        reset_btn.click(fn=processor.reset_conversation)

    # Launch the app
    demo.launch()

if __name__ == "__main__":
    main() 