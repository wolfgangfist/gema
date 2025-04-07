import asyncio
import os
import time
import threading
import json
import queue
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any, Optional, Callable
from generator import Generator, Segment, load_csm_1b_local, generate_streaming_audio
from llm_interface import LLMInterface
from rag_system import RAGSystem 
from vad import AudioStreamProcessor, VoiceActivityDetector
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine("sqlite:///companion.db")
SessionLocal = sessionmaker(bind=engine)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(String)
    user_message = Column(Text)
    ai_message = Column(Text)
    audio_path = Column(String)

Base.metadata.create_all(bind=engine)

# Pydantic models for API
class CompanionConfig(BaseModel):
    system_prompt: str
    reference_audio_path: str
    reference_text: str
    model_path: str
    llm_path: str
    max_tokens: int = 8192
    voice_speaker_id: int = 0
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    embedding_model: str = "all-MiniLM-L6-v2"

# Application state
conversation_history = []
config = None
audio_queue = queue.Queue()
is_speaking = False
interrupt_flag = False
generator = None
llm = None
rag = None
vad_processor = None
active_connections = []  # List to store active WebSocket connections
message_queue = asyncio.Queue()  # Queue for passing messages to async context

# Async event loop for background tasks
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Helper function to send messages to all clients
async def send_to_all_clients(message):
    """Send a message to all connected clients"""
    for client in active_connections[:]:  # Create a copy of the list to allow modification during iteration
        try:
            if client.client_state == 1:  # WebSocket.OPEN
                await client.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            # Remove broken connections
            if client in active_connections:
                active_connections.remove(client)

# Helper to send transcription to a specific client
async def send_transcription(websocket, text):
    """Send transcription to client if websocket is still open"""
    try:
        if websocket.client_state == 1:  # WebSocket.OPEN
            await websocket.send_json({"type": "transcription", "text": text})
    except Exception as e:
        logger.error(f"Error sending transcription: {e}")

# Background task to process message queue
async def process_message_queue():
    """Process messages from the queue and send to clients"""
    while True:
        message = await message_queue.get()
        await send_to_all_clients(message)
        message_queue.task_done()

def initialize_models(config_data):
    global generator, llm, rag, vad_processor
    
    logger.info("Loading voice model...")
    start_time = time.time()
    generator = load_csm_1b_local(config_data.model_path, "cuda")
    logger.info(f"Voice model loaded in {time.time() - start_time:.2f} seconds")
    
    logger.info("Loading LLM...")
    start_time = time.time()
    llm = LLMInterface(config_data.llm_path, config_data.max_tokens)
    logger.info(f"LLM loaded in {time.time() - start_time:.2f} seconds")
    
    logger.info(f"Initializing enhanced RAG system with model {config_data.embedding_model}...")
    rag = RAGSystem("companion.db", model_name=config_data.embedding_model)
    
    # Load Silero VAD model
    logger.info("Loading Silero VAD model...")
    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
        )
        
        logger.info("Silero VAD model loaded successfully")
        
        # Get VAD utils
        get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = vad_utils
        
        logger.info("Initializing VAD processor...")
        vad_processor = AudioStreamProcessor(
            model=vad_model, 
            utils=vad_utils, 
            sample_rate=16000,
            vad_threshold=config_data.vad_threshold,
            callbacks={
                "on_speech_start": on_speech_start,
                "on_speech_end": on_speech_end
            }
        )
        logger.info("VAD processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize VAD processor: {e}")
        raise
    
    # Load reference audio
    if os.path.exists(config_data.reference_audio_path):
        load_reference_segment(
            config_data.reference_audio_path, 
            config_data.reference_text,
            config_data.voice_speaker_id
        )
    else:
        logger.warning(f"Reference audio not found at {config_data.reference_audio_path}")

## VAD callback functions
def on_speech_start():
    logger.info("VAD detected speech start")
    # Put message in queue for async processing
    try:
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "vad_status", "status": "speech_started"}),
            loop
        )
        logger.info("Speech start event queued successfully")
    except Exception as e:
        logger.error(f"Error queueing speech start event: {e}")

def on_speech_end(audio_data, sample_rate):
    logger.info(f"VAD detected speech end, processing audio of length {len(audio_data)}...")
    
    try:
        # Process the complete audio segment
        # Wrap transcription in a try-except block since this is where errors occur
        try:
            user_text = transcribe_audio(audio_data, sample_rate)
            logger.info(f"Transcribed text: {user_text}")
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            import traceback
            logger.error(traceback.format_exc())
            user_text = "Sorry, I couldn't transcribe that properly."
        
        # Put message in queue for async processing
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "transcription", "text": user_text}),
            loop
        )
        logger.info("Transcription queued successfully")
        
        # Process response in a separate thread to avoid blocking
        threading.Thread(
            target=lambda: process_user_input(user_text),
            daemon=True
        ).start()
        
    except Exception as e:
        logger.error(f"Error in speech end processing: {e}")
        import traceback
        logger.error(traceback.format_exc())

def load_reference_segment(audio_path, text, speaker_id=0):
    global reference_segments
    logger.info(f"Loading reference audio: {audio_path}")
    start_time = time.time()
    
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    
    reference_segments = [
        Segment(text=text, speaker=speaker_id, audio=audio_tensor)
    ]
    
    logger.info(f"Reference audio loaded in {time.time() - start_time:.2f} seconds")

# Audio generation in separate thread
def audio_generation_thread(text, output_file):
    global is_speaking, interrupt_flag, audio_queue
    
    is_speaking = True
    interrupt_flag = False
    
    try:
        # Create directory for output file if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Define stream callback to send audio to the queue
        def stream_callback(audio_chunk):
            if interrupt_flag:
                return False  # Stop generation
            
            # Convert to float32 numpy array for sounddevice
            audio_np = audio_chunk.cpu().numpy().astype(np.float32)
            audio_queue.put(audio_np)
            return True  # Continue generation
        
        # Open an audio file to write the final result
        all_audio_chunks = []
        
        # Use the low-level generator.generate_stream API
        for audio_chunk in generator.generate_stream(
            text=text,
            speaker=config.voice_speaker_id,
            context=reference_segments,
            max_audio_length_ms=40_000,
            temperature=0.7,
            topk=30
        ):
            # Check if interrupted
            if interrupt_flag:
                break
                
            # Handle the audio chunk
            all_audio_chunks.append(audio_chunk)
            
            # Process for real-time playback
            stream_callback(audio_chunk)
        
        # Save the complete audio
        if all_audio_chunks:
            complete_audio = torch.cat(all_audio_chunks)
            torchaudio.save(output_file, complete_audio.unsqueeze(0), generator.sample_rate)
    
    except Exception as e:
        logger.error(f"Error in audio generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        is_speaking = False
        # Signal end of audio
        audio_queue.put(None)
        
        # Notify clients that audio generation is complete
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "audio_status", "status": "complete"}),
            loop
        )
# Audio playback thread
def audio_playback_thread():
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            continue  # End of current audio
        
        # Play audio
        sd.play(audio_chunk, generator.sample_rate)
        sd.wait()

# Initialize Whisper for speech recognition
whisper_model = whisper.load_model("base")

def transcribe_audio(audio_data, sample_rate):
    """Transcribe audio data with error handling for PyTorch graph issues"""
    global whisper_model
    
    # Convert audio to format expected by Whisper
    audio_np = np.array(audio_data).astype(np.float32)
    
    # Resample if needed
    if sample_rate != 16000:
        try:
            # Implement resampling here using torchaudio
            audio_tensor = torch.tensor(audio_np).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=16000
            )
            audio_np = audio_tensor.squeeze(0).numpy()
        except Exception as e:
            logger.error(f"Error during resampling: {e}")
            # If resampling fails, use the original audio (might cause issues, but better than failing)
    
    # Save a copy of the audio data for debugging (optional)
    try:
        import os
        debug_dir = "debug_audio"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        debug_file = os.path.join(debug_dir, f"debug_audio_{timestamp}.npy")
        np.save(debug_file, audio_np)
        logger.info(f"Saved debug audio to {debug_file}")
    except Exception as e:
        logger.error(f"Error saving debug audio: {e}")
    
    # Transcribe with Whisper
    try:
        # Disable Pytorch graph capture for this operation
        with torch.jit.optimized_execution(False):
            # Set to CPU explicitly to avoid graph capture issues
            if hasattr(whisper_model, "to"):
                whisper_model.to("cpu")
            
            # Transcribe with additional options to avoid graph optimization issues
            result = whisper_model.transcribe(
                audio_np,
                fp16=False  # Avoid half precision which might trigger graph capture
            )
            
            return result["text"]
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback transcription with even more basic settings
        try:
            logger.info("Attempting fallback transcription...")
            # Try to reload the model if necessary
            whisper_model = whisper.load_model("base", device="cpu")
            
            # Basic transcription with minimal options
            result = whisper_model.transcribe(audio_np, language="en")
            return result["text"]
        except Exception as e2:
            logger.error(f"Fallback transcription also failed: {e2}")
            return "[Transcription error]"
        
# Process user input and generate response
def process_user_input(user_text, client=None, session_id="default"):
    logger.info(f"Processing user input: {user_text[:50]}...")
    
    # Get response from LLM with RAG
    context = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" 
                        for msg in conversation_history[-5:]])
    
    rag_context = rag.query(user_text)
    system_prompt = config.system_prompt
    
    if rag_context:
        logger.info("Found relevant context from RAG")
        system_prompt += f"\n\nRelevant context from previous conversations:\n{rag_context}"
    
    ai_response = llm.generate_response(
        system_prompt=system_prompt,
        user_message=user_text,
        conversation_history=context
    )
    
    # Store in conversation history
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    conversation_history.append({
        "timestamp": timestamp,
        "user": user_text,
        "ai": ai_response
    })
    
    # Store in database
    db = SessionLocal()
    db_conversation = Conversation(
        session_id=session_id,
        timestamp=timestamp,
        user_message=user_text,
        ai_message=ai_response,
        audio_path=""  # Will be updated after generation
    )
    db.add(db_conversation)
    db.commit()
    
    # Get conversation ID for RAG indexing
    conv_id = db_conversation.id
    
    # Send text response to all clients
    asyncio.run_coroutine_threadsafe(
        message_queue.put({"type": "response", "text": ai_response}),
        loop
    )
    
    # Generate audio response in a separate thread
    output_file = f"responses/{timestamp.replace(':', '-')}.wav"
    os.makedirs("responses", exist_ok=True)
    
    # Update database with audio path
    db_conversation.audio_path = output_file
    db.commit()
    
    # Add to RAG system asynchronously - FIXED: don't pass conv_id
    threading.Thread(
        target=lambda: rag.add_conversation(user_text, ai_response),
        daemon=True
    ).start()
    
    db.close()
    
    # Start audio generation
    threading.Thread(
        target=audio_generation_thread,
        args=(ai_response, output_file),
        daemon=True
    ).start()
    
    # Send notification that audio is being generated
    asyncio.run_coroutine_threadsafe(
        message_queue.put({"type": "audio_status", "status": "generating"}),
        loop
    )
    
    return ai_response

# FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Add to active connections
    active_connections.append(websocket)
    logger.info(f"New client connected. Total clients: {len(active_connections)}")
    
    # Start audio playback thread if not already running
    if not any(thread.name == "audio_playback" for thread in threading.enumerate()):
        playback_thread = threading.Thread(
            target=audio_playback_thread, 
            daemon=True, 
            name="audio_playback"
        )
        playback_thread.start()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data["type"] == "config":
                # Update configuration
                global config
                config = CompanionConfig(**data["config"])
                initialize_models(config)
                await websocket.send_json({"type": "status", "message": "Models initialized"})
            
            elif data["type"] == "audio":
                logger.info("Processing incoming audio")
                audio_data = np.array(data["audio"]).astype(np.float32)
                sample_rate = data["sample_rate"]
                session_id = data.get("session_id", "default")
                
                if config and config.vad_enabled:
                    logger.info("Processing with VAD")
                    vad_processor.process_audio(audio_data)
                else:
                    logger.info("Traditional approach - transcribe immediately")
                    user_text = transcribe_audio(audio_data, sample_rate)
                    await websocket.send_json({"type": "transcription", "text": user_text})
                    process_user_input(user_text, websocket, session_id)
            
            elif data["type"] == "interrupt":
                # Handle interruption
                if is_speaking:
                    interrupt_flag = True
                    audio_queue = queue.Queue()  # Clear the queue
                    await websocket.send_json({"type": "audio_status", "status": "interrupted"})
                
            elif data["type"] == "mute":
                # Handle mute/unmute
                muted = data["muted"]
                await websocket.send_json({"type": "mute_status", "muted": muted})
                
                # If unmuting, reset VAD to start fresh
                if not muted and config and config.vad_enabled:
                    vad_processor.reset()
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        # Remove from active connections
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"Client disconnected. Remaining clients: {len(active_connections)}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Remove from active connections
        if websocket in active_connections:
            active_connections.remove(websocket)

# Homepage
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    global loop
    
    # Create necessary directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("responses", exist_ok=True)
    os.makedirs("embeddings_cache", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Create index.html template that redirects to static/index.html
    template_dir = os.path.join(os.getcwd(), "templates")
    with open(os.path.join(template_dir, "index.html"), "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=/static/index.html">
</head>
<body>
    <p>Redirecting to <a href="/static/index.html">AI Companion</a>...</p>
</body>
</html>
        """)
    
    # Download Silero VAD model
    # Note: We don't initialize AudioStreamProcessor here anymore
    # We'll just pre-download the model
    import torch

    try:
        logger.info("Pre-downloading Silero VAD model from torch.hub...")
        torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        logger.info("Silero VAD model pre-downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-download Silero VAD model: {e}")

    # Start background task for processing message queue
    asyncio.create_task(process_message_queue())
    logger.info("Background task for message queue started")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down server...")

# Start the server
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Companion server")
    
    # Start the message queue processing thread
    threading.Thread(
        target=lambda: asyncio.run(loop.run_forever()),
        daemon=True,
        name="asyncio_loop"
    ).start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)