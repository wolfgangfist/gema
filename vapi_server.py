import io
import time
import numpy as np
from typing import Generator as PyGenerator, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from model_handler import VoiceGenerator


app = FastAPI()
voice_generator = VoiceGenerator()
voice_generator.initialize()


@app.post('/custom-voice-tts')
async def handle_custom_voice_request(request: Request):
    """Handle TTS requests with streaming or non-streaming output."""
    request_start_time = time.monotonic()
    target_sr = 16000  # Target telephony rate

    try:
        data = await request.json()
        message_data = data.get('message', {})
        text_to_speak = message_data.get('text')
        requested_sample_rate = message_data.get('sampleRate', target_sr)
        stream_mode = message_data.get('stream', False)  # Toggle streaming
        call_id = message_data.get('call', {}).get('id', 'N/A')
        speaker_id = 0  # Default speaker

        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} | Call ID: {call_id}")
        print(f"Text: \"{text_to_speak[:150]}{'...' if text_to_speak and len(text_to_speak) > 150 else ''}\"")
        print(f"Mode: {'Streaming' if stream_mode else 'Non-streaming'}")

        if not text_to_speak:
            print("ERROR: No text provided.")
            return StreamingResponse(iter([b'']), status_code=400, media_type=f'audio/l16; rate={requested_sample_rate}; channels=1')

        print(f"Starting generation. Handler setup: {(time.monotonic() - request_start_time) * 1000:.2f} ms.")
        
        if stream_mode:
            # Streaming response
            pcm_stream = voice_generator.generate_pcm_stream(
                text=text_to_speak,
                speaker_id=speaker_id,
                target_sample_rate=requested_sample_rate,
                max_audio_length_ms=30_000,
                temperature=0.9,
                topk=50,
                stream_chunk_frames=10
            )
            return StreamingResponse(
                pcm_stream,
                media_type=f'audio/l16; rate={requested_sample_rate}; channels=1'
            )
        else:
            # Non-streaming response
            pcm_bytes = voice_generator.generate_pcm_full(
                text=text_to_speak,
                speaker_id=speaker_id,
                target_sample_rate=requested_sample_rate,
                max_audio_length_ms=30_000,
                temperature=0.9,
                topk=50
            )
            return Response(
                content=pcm_bytes,
                media_type=f'audio/l16; rate={requested_sample_rate}; channels=1'
            )

    except RuntimeError as e:
        if str(e) == "Generator not initialized.":
            print("ERROR: Generator not initialized.")
            return StreamingResponse(iter([b'']), status_code=503, media_type=f'audio/l16; rate={requested_sample_rate}; channels=1')
        print(f"Error generating audio: {e}")
        return StreamingResponse(iter([b'']), status_code=500, media_type=f'audio/l16; rate={requested_sample_rate}; channels=1')
    except Exception as e:
        print(f"Server error: {e}")
        return StreamingResponse(iter([b'']), status_code=500, media_type=f'audio/l16; rate={requested_sample_rate}; channels=1')

@app.get('/ping')
async def ping():
    """Health check endpoint indicating model status."""
    if voice_generator.generator is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Model ready"}