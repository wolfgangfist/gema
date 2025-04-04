import os
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastrtc import Stream, ReplyOnPause
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment

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
        return {"status": "success", "message": "Conversation reset"}

# Initialize processor
processor = CSMSpeechProcessor()

# Create FastRTC stream
stream = Stream(
    handler=ReplyOnPause(processor.process_speech),
    modality="audio",
    mode="send-receive",
)

# Create FastAPI app
app = FastAPI(title="CSM Speech-to-Speech API")

# Mount the stream on the FastAPI app
stream.mount(app)

@app.get("/")
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CSM Speech-to-Speech</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            p {
                color: #666;
            }
            .button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CSM Speech-to-Speech Conversational AI</h1>
            <p>This API provides speech-to-speech functionality using the CSM model from Sesame.</p>
            <p>For the interactive UI version, visit: <a href="/ui">/ui</a></p>
            <p>API endpoints:</p>
            <ul>
                <li><a href="/docs">/docs</a> - API documentation</li>
                <li><a href="/reset">/reset</a> - Reset conversation</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/reset")
async def reset_conversation():
    return processor.reset_conversation()

# Mount UI at /ui path
@app.get("/ui")
async def ui():
    html_content = stream.ui.generate_html()
    return HTMLResponse(content=html_content)

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 