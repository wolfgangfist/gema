# CSM Speech-to-Speech with FastRTC

This implementation enables end-to-end speech-to-speech conversations using the CSM (Conversational Speech Model) from Sesame and FastRTC for real-time communication.

## Features

- **End-to-End Speech Processing**: Direct speech-to-speech without intermediate text representation
- **Real-Time Communication**: Using WebRTC via FastRTC
- **Turn-Taking**: Automatic pause detection for natural conversation flow
- **Multiple Deployment Options**: Gradio UI, FastAPI, or telephone access
- **Voice Identity Preservation**: Maintains consistent voice characteristics

## Installation

1. Clone the repository and navigate to the speech-to-speech-fastrtc branch:
   ```bash
   git clone https://github.com/SesameAILabs/csm.git
   cd csm
   git checkout speech-to-speech-fastrtc
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Gradio UI

Run the Gradio interface for an interactive web UI:

```bash
python speech_to_speech.py
```

This will launch a local web server with the Gradio UI. Open your browser to the displayed URL.

### Option 2: FastAPI Deployment

Run the FastAPI server for a more production-ready deployment:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Access the UI at http://localhost:8000/ui or the API at http://localhost:8000/docs

### Option 3: Telephone Access

Use FastRTC's fastphone feature to get a temporary phone number:

```python
from fastrtc import Stream, ReplyOnPause
from app import processor

# Create stream with the same processor
stream = Stream(
    handler=ReplyOnPause(processor.process_speech),
    modality="audio",
    mode="send-receive"
)

# Get a temporary phone number
phone_info = stream.fastphone()
print(f"Call this number to interact with CSM: {phone_info['phone_number']}")
```

## Deployment on Hugging Face

### Spaces Deployment

1. Create a new Hugging Face Space
2. Use the Gradio SDK
3. Upload all files from this branch
4. The Space will automatically use `app.py`

### Inference Endpoints

1. Create a new Inference Endpoint using the Automatic serving framework
2. Point to your repository with this branch
3. The endpoint will use `app.py` as the entry point

## How It Works

1. **Audio Capture**: FastRTC handles capturing audio from the user
2. **Turn Detection**: ReplyOnPause detects when the user stops speaking
3. **Audio Processing**: CSM processes the speech directly through Mimi's tokenization
4. **Response Generation**: CSM generates a spoken response based on the conversation context
5. **Audio Playback**: FastRTC handles playing the audio response to the user

## Architecture

- **FastRTC**: Handles WebRTC communication, turn-taking, and UI
- **CSM**: Processes speech and generates responses
- **Gradio/FastAPI**: Provides web interfaces

## License

This implementation is released under the same license as the CSM project from Sesame. 