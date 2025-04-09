import asyncio
import os
import sqlite3
import time
import threading
import json
import queue
from fastapi.websockets import WebSocketState
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
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
from config import ConfigManager
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
speaker_counters = {
    0: 0,  # AI
    1: 0   # User
}
pending_user_inputs = []
user_input_lock = threading.Lock()
audio_fade_duration = 0.3  # seconds for fade-out
last_interrupt_time = 0
interrupt_cooldown = 1.0  # seconds between allowed interrupts
audio_chunk_buffer = []  # Buffer to store the most recent audio chunks for fade-out
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
llm_lock = threading.Lock()
audio_gen_lock = threading.Lock()
# Database
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

# Pydantic config schema
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

# Global state
conversation_history = []
config = None
audio_queue = queue.Queue()
is_speaking = False
interrupt_flag = False
generator = None
llm = None
rag = None
vad_processor = None
reference_segments = []
active_connections = []
message_queue = asyncio.Queue()

# Async event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
config_manager = ConfigManager()
model_id = "openai/whisper-large-v3-turbo"
# Whisper
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
)

processor = AutoProcessor.from_pretrained(model_id)
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device='cuda',
)
# Background queue
async def process_message_queue():
    while True:
        message = await message_queue.get()
        for client in active_connections[:]:
            try:
                if client.client_state == 1:
                    await client.send_json(message)
            except:
                if client in active_connections:
                    active_connections.remove(client)
        message_queue.task_done()

def load_reference_segment(audio_path, text, speaker_id=0):
    global reference_segments
    logger.info(f"Loading reference audio: {audio_path}")
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    reference_segments = [Segment(text=text, speaker=speaker_id, audio=audio_tensor)]
    logger.info(f"Reference audio loaded")

def transcribe_audio(audio_data, sample_rate):
    global whisper_model
    audio_np = np.array(audio_data).astype(np.float32)
    if sample_rate != 16000:
        try:
            audio_tensor = torch.tensor(audio_np).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
            audio_np = audio_tensor.squeeze(0).numpy()
        except: pass
    try:
        with torch.jit.optimized_execution(False):
            whisper_model.to("cuda")
            result = whisper_pipe(audio_np)
            whisper_model.to("cpu")
            return result["text"]
    except:
        return "[Transcription error]"

def initialize_models(config_data):
    global generator, llm, rag, vad_processor, config
    config = config_data

    logger.info("Loading voice model...")
    generator = load_csm_1b_local(config_data.model_path, "cuda")
    logger.info("Loading LLM...")
    llm = LLMInterface(config_data.llm_path, config_data.max_tokens)
    logger.info("Loading RAG...")
    rag = RAGSystem("companion.db", model_name=config_data.embedding_model)

    vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    vad_processor = AudioStreamProcessor(
        model=vad_model, 
        utils=vad_utils, 
        sample_rate=16000,
        vad_threshold=config_data.vad_threshold,
        callbacks={"on_speech_start": on_speech_start, "on_speech_end": on_speech_end}
    )

    if os.path.exists(config_data.reference_audio_path):
        load_reference_segment(config_data.reference_audio_path, config_data.reference_text, config_data.voice_speaker_id)

def on_speech_start():
    asyncio.run_coroutine_threadsafe(
        message_queue.put({"type": "vad_status", "status": "speech_started"}),
        loop
    )

def on_speech_end(audio_data, sample_rate):
    try:
        logger.info("Transcription starting")
        user_text = transcribe_audio(audio_data, sample_rate)
        logger.info(f"Transcription completed: '{user_text}'")

        session_id = "default"
        speaker_id = 1
        index = speaker_counters[speaker_id]
        user_audio_path = f"audio/user/{session_id}_user_{index}.wav"
        os.makedirs(os.path.dirname(user_audio_path), exist_ok=True)

        audio_tensor = torch.tensor(audio_data).unsqueeze(0)
        save_audio_and_trim(user_audio_path, session_id, speaker_id, audio_tensor.squeeze(0), sample_rate)
        add_segment(user_text, speaker_id, audio_tensor.squeeze(0))

        logger.info(f"User audio saved and segment appended: {user_audio_path}")

        speaker_counters[speaker_id] += 1

        send_to_all_clients({"type": "transcription", "text": user_text})
        threading.Thread(target=lambda: process_user_input(user_text, session_id), daemon=True).start()
    except Exception as e:
        logger.error(f"VAD callback failed: {e}")

def process_pending_inputs():
    """Process any pending user inputs after an interruption"""
    global pending_user_inputs
    
    with user_input_lock:
        if not pending_user_inputs:
            return
            
        # If we have multiple inputs, combine them
        if len(pending_user_inputs) > 1:
            combined_text = " ".join([text for text, _ in pending_user_inputs])
            logger.info(f"Combining {len(pending_user_inputs)} inputs: '{combined_text}'")
            
            # Use the session_id from the first input
            session_id = pending_user_inputs[0][1]
            pending_user_inputs = []
            
            # Process the combined input
            threading.Thread(target=lambda: process_user_input(combined_text, session_id), daemon=True).start()
        else:
            # Just process the single input
            user_text, session_id = pending_user_inputs[0]
            pending_user_inputs = []
            threading.Thread(target=lambda: process_user_input(user_text, session_id), daemon=True).start()


def handle_interrupt(websocket):
    global is_speaking, interrupt_flag, audio_queue, last_interrupt_time
    
    # Implement interrupt cooldown to prevent rapid interruptions
    current_time = time.time()
    if current_time - last_interrupt_time < interrupt_cooldown:
        logger.info("Ignoring interrupt: too soon after previous interrupt")
        return False
        
    last_interrupt_time = current_time
    
    if is_speaking:
        logger.info("Interruption requested")
        interrupt_flag = True
        
        # Reset VAD to prepare for new input
        vad_processor.reset()
        
        # Notify client of interruption
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "audio_status", "status": "interrupted"}),
            loop
        )
        return True
    return False


def process_user_input(user_text, session_id="default"):
    """Modified to handle multiple inputs during interruption"""
    global config, is_speaking, pending_user_inputs
    
    if is_speaking:
        with user_input_lock:
            pending_user_inputs.append((user_text, session_id))
            logger.info(f"Added user input to pending queue: '{user_text}'")
        return
    
    context = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in conversation_history[-5:]])
    rag_context = rag.query(user_text)
    system_prompt = config.system_prompt
    if rag_context:
        system_prompt += f"\n\nRelevant context:\n{rag_context}"
    
    torch.cuda.empty_cache()

    send_to_all_clients({"type": "status", "message": "Thinking..."})
    
    try:
        with llm_lock: 
            ai_response = llm.generate_response(system_prompt, user_text, context)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        conversation_history.append({
            "timestamp": timestamp,
            "user": user_text,
            "ai": ai_response
        })
        
        try:
            db = SessionLocal()
            conv = Conversation(
                session_id=session_id,
                timestamp=timestamp,
                user_message=user_text,
                ai_message=ai_response,
                audio_path=""
            )
            db.add(conv)
            db.commit()
            index = speaker_counters[0]
            output_file = f"audio/ai/{session_id}_response_{index}.wav"
            speaker_counters[0] += 1
            conv.audio_path = output_file
            db.commit()
            db.close()
        except Exception as e:
            logger.error(f"Database error: {e}")
        
        threading.Thread(target=lambda: rag.add_conversation(user_text, ai_response), daemon=True).start()
        
        send_to_all_clients({"type": "response", "text": ai_response})

        time.sleep(0.5)
        
        torch.cuda.empty_cache()
        
        threading.Thread(target=audio_generation_thread, args=(ai_response, output_file), daemon=True).start()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "error", "message": "Failed to generate response"}),
            loop
        )

def send_to_all_clients(message: dict):
    for client in active_connections[:]:
        try:
            if client.application_state == WebSocketState.CONNECTED:
                asyncio.run_coroutine_threadsafe(client.send_json(message), loop)
                logger.info(f"Sent message to client: {message}")
            else:
                logger.warning("Detected non-connected client; removing from active_connections")
                active_connections.remove(client)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            if client in active_connections:
                active_connections.remove(client)

saved_audio_paths = {
    "default": {
        0: [],  # AI
        1: []   # User
    }
}
MAX_AUDIO_FILES = 10

def save_audio_and_trim(path, session_id, speaker_id, tensor, sample_rate):
    """
    Save audio file and trim old audio files for both AI and user to maintain storage limits.
    
    Args:
        path: Path to save the audio file
        session_id: Conversation session ID
        speaker_id: 0 for AI, 1 for user
        tensor: Audio tensor to save
        sample_rate: Audio sample rate
    """
    torchaudio.save(path, tensor.unsqueeze(0), sample_rate)
    
    saved_audio_paths.setdefault(session_id, {}).setdefault(speaker_id, []).append(path)
    
    paths = saved_audio_paths[session_id][speaker_id]
    while len(paths) > MAX_AUDIO_FILES:
        old_path = paths.pop(0)
        if os.path.exists(old_path):
            os.remove(old_path)
            logger.info(f"Removed old audio file: {old_path}")
    
    other_speaker_id = 1 if speaker_id == 0 else 0
    if other_speaker_id in saved_audio_paths[session_id]:
        other_paths = saved_audio_paths[session_id][other_speaker_id]
        while len(other_paths) > MAX_AUDIO_FILES:
            old_path = other_paths.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.info(f"Removed old audio file from other speaker: {old_path}")

MAX_SEGMENTS = 10

def add_segment(text, speaker_id, audio_tensor):
    reference_segments.append(Segment(text=text, speaker=speaker_id, audio=audio_tensor))
    while len(reference_segments) > MAX_SEGMENTS:
        reference_segments.pop(0)

def fade_out_audio():
    """Apply a smooth fade-out effect to the buffered audio chunks and send to output"""
    global audio_chunk_buffer, audio_queue
    
    if not audio_chunk_buffer:
        return
        
    # Concatenate all chunks in the buffer
    combined = np.concatenate(audio_chunk_buffer)
    
    # Use a cosine (Hann) window for smoother fade-out
    # This creates a more natural-sounding fade than linear
    fade_samples = len(combined)
    fade_window = np.cos(np.linspace(0, np.pi/2, fade_samples))**2
    fade_envelope = fade_window[::-1]  # Reverse to get fade-out
    
    # Apply fade-out with appropriate normalization to prevent clipping
    max_val = np.max(np.abs(combined))
    if max_val > 0:
        # Normalize to prevent clipping
        normalized = combined / max_val
        faded_audio = normalized * fade_envelope
        # Add a short period of silence at the end to prevent clicks
        silence_samples = int(0.05 * generator.sample_rate)  # 50ms silence
        silence = np.zeros(silence_samples)
        faded_audio = np.concatenate([faded_audio, silence])
    else:
        faded_audio = combined * fade_envelope
    
    # Apply a low-pass filter to smooth any discontinuities
    # This is a simple moving average filter
    window_size = min(32, len(faded_audio) // 10)
    if window_size > 1:
        window = np.ones(window_size) / window_size
        # Only filter the end of the audio where the fade happens
        end_section = faded_audio[-window_size*4:]
        filtered_end = np.convolve(end_section, window, mode='same')
        faded_audio[-window_size*4:] = filtered_end
    
    # Send the faded audio to the output queue
    audio_queue.put(faded_audio)
    
    # Clear the buffer
    audio_chunk_buffer = []

def audio_playback_thread():
    """
    Improved audio playback thread that prevents clipping between audio chunks.
    Uses a single continuous buffer for smooth playback.
    """
    global audio_queue
    
    # Create a more efficient playback system
    playback_buffer = np.array([], dtype=np.float32)
    
    while True:
        try:
            # Get next chunk (with timeout to check for program exit)
            chunk = audio_queue.get(timeout=0.1)
            
            if chunk is None:
                # If we get None, it means current generation is complete
                # Play any remaining audio in buffer
                if len(playback_buffer) > 0:
                    sd.play(playback_buffer, generator.sample_rate)
                    sd.wait()
                    playback_buffer = np.array([], dtype=np.float32)
                continue
                
            # Add chunk to buffer (no crossfade needed for normal playback)
            playback_buffer = np.concatenate([playback_buffer, chunk]) if len(playback_buffer) > 0 else chunk
            
            # Play when buffer has enough data for efficiency
            # Avoid playing tiny chunks which can cause gaps
            min_buffer_time = 0.1  # seconds
            min_buffer_samples = int(min_buffer_time * generator.sample_rate)
            
            if len(playback_buffer) >= min_buffer_samples:
                sd.play(playback_buffer, generator.sample_rate)
                sd.wait()
                playback_buffer = np.array([], dtype=np.float32)
                
        except queue.Empty:
            # Queue is empty but we have data in buffer - play it
            if len(playback_buffer) > 0:
                sd.play(playback_buffer, generator.sample_rate)
                sd.wait()
                playback_buffer = np.array([], dtype=np.float32)
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            # Clear buffer on error to avoid getting stuck
            playback_buffer = np.array([], dtype=np.float32)


def audio_generation_thread(text, output_file):
    """
    Modified audio generation thread that doesn't introduce clipping.
    Simplifies buffer handling for normal audio playback.
    """
    global is_speaking, interrupt_flag, audio_queue
    
    # Try to acquire the lock, but don't block if already locked
    if not audio_gen_lock.acquire(blocking=False):
        logger.warning("Another audio generation is in progress, skipping this one")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "error", "message": "Audio generation busy, skipping synthesis"}),
            loop
        )
        return
    
    try:
        is_speaking = True
        interrupt_flag = False
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        all_audio_chunks = []
        
        try:
            # Clear CUDA cache before starting
            torch.cuda.empty_cache()

            for audio_chunk in generator.generate_stream(
                text=text, speaker=config.voice_speaker_id, context=reference_segments,
                max_audio_length_ms=40000, temperature=0.7, topk=30
            ):
                if interrupt_flag:
                    # For interruption, implement a simple fade out
                    # This only happens during interruptions, not normal playback
                    if len(all_audio_chunks) > 0:
                        # Get last chunk and apply a simple fade out
                        last_chunk = audio_chunk.cpu().numpy().astype(np.float32)
                        fade_samples = min(len(last_chunk), int(0.1 * generator.sample_rate))  # 100ms fade
                        if fade_samples > 0:
                            fade = np.linspace(1.0, 0.0, fade_samples)
                            last_chunk[-fade_samples:] *= fade
                            audio_queue.put(last_chunk)
                    break
                    
                all_audio_chunks.append(audio_chunk)
                
                # Send chunk directly to audio queue without any buffering
                audio_queue.put(audio_chunk.cpu().numpy().astype(np.float32))
                
            if all_audio_chunks and not interrupt_flag:
                complete_audio = torch.cat(all_audio_chunks)
                save_audio_and_trim(output_file, "default", config.voice_speaker_id, complete_audio, generator.sample_rate)
                add_segment(text, config.voice_speaker_id, complete_audio)

        except RuntimeError as e:
            logger.error(f"CUDA memory error during audio generation: {e}")
            audio_queue.put(None)
            asyncio.run_coroutine_threadsafe(
                message_queue.put({"type": "error", "message": "Audio generation failed due to GPU memory error"}),
                loop
            )
        finally:
            is_speaking = False
            audio_queue.put(None)  # Signal end of generation
            asyncio.run_coroutine_threadsafe(
                message_queue.put({"type": "audio_status", "status": "complete"}),
                loop
            )
            
            # Process any pending user inputs
            with user_input_lock:
                if pending_user_inputs:
                    user_text, session_id = pending_user_inputs.pop(0)
                    pending_user_inputs.clear()  # Clear any other pending inputs
                    threading.Thread(target=lambda: process_user_input(user_text, session_id), daemon=True).start()
    finally:
        # Always release the lock
        audio_gen_lock.release()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global last_interrupt_time, is_speaking, interrupt_flag, audio_queue
    
    await websocket.accept()
    active_connections.append(websocket)
    
    # Initialize audio playback thread with standard implementation
    if not any(t.name == "audio_playback" for t in threading.enumerate()):
        threading.Thread(target=audio_playback_thread, daemon=True, name="audio_playback").start()
        
    saved = config_manager.load_config()
    if saved:
        await websocket.send_json({"type": "saved_config", "config": saved})
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "config":
                # Config code unchanged
                conf = CompanionConfig(**data["config"])
                config_manager.save_config(data["config"])
                initialize_models(conf)
                await websocket.send_json({"type": "status", "message": "Models initialized and configuration saved"})
            elif data["type"] == "request_saved_config":
                saved = config_manager.load_config()
                await websocket.send_json({"type": "saved_config", "config": saved})
            elif data["type"] == "audio":
                # Audio handling code
                audio_data = np.array(data["audio"]).astype(np.float32)
                sample_rate = data["sample_rate"]
                
                if sample_rate != 16000:
                    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                    audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
                    audio_data = audio_tensor.squeeze(0).numpy()
                    sample_rate = 16000
                
                if config and config.vad_enabled:
                    vad_processor.process_audio(audio_data)
                else:
                    text = transcribe_audio(audio_data, sample_rate)
                    print(f"transcript: {text}")
                    await websocket.send_json({"type": "transcription", "text": text})
                    await message_queue.put({"type": "transcription", "text": text})
                    
                    if is_speaking:
                        with user_input_lock:
                            pending_user_inputs.append((text, "default"))
                            logger.info(f"Added user input to pending queue: '{text}'")
                        
                        # Simple interrupt handling
                        interrupt_flag = True
                        vad_processor.reset()
                        await websocket.send_json({"type": "audio_status", "status": "interrupted"})
                    else:
                        process_user_input(text)
            elif data["type"] == "interrupt":
                # Simple interrupt handling - just set the flag and reset
                if is_speaking:
                    interrupt_flag = True
                    vad_processor.reset()
                    await websocket.send_json({"type": "audio_status", "status": "interrupted"})
            elif data["type"] == "mute":
                await websocket.send_json({"type": "mute_status", "muted": data["muted"]})
                if not data["muted"] and config and config.vad_enabled:
                    vad_processor.reset()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    return templates.TemplateResponse("setup.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    os.makedirs("static", exist_ok=True)
    os.makedirs("audio/user", exist_ok=True)
    os.makedirs("audio/ai", exist_ok=True)
    os.makedirs("embeddings_cache", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    with open("templates/index.html", "w") as f:
        f.write("""<meta http-equiv="refresh" content="0; url=/setup" />""")
    try:
        torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    except: pass
    asyncio.create_task(process_message_queue())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutting down...")

from flask import Flask, jsonify, request, send_file

@app.get("/api/conversations")
async def get_conversations(request: Request):
    conn = sqlite3.connect("companion.db")
    cur = conn.cursor()
    cur.execute("SELECT id, user_message, ai_message FROM conversations ORDER BY id DESC")
    data = [{"id": row[0], "user_message": row[1], "ai_message": row[2]} for row in cur.fetchall()]
    conn.close()
    return JSONResponse(content=data)

@app.route("/api/conversations/<int:conv_id>", methods=["PUT"])
def update_conversation(conv_id):
    data = request.get_json()
    conn = sqlite3.connect("companion.db")
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET user_message=?, ai_message=? WHERE id=?",
                (data["user_message"], data["ai_message"], conv_id))
    conn.commit()
    conn.close()
    return "", 204

@app.delete("/api/conversations")
async def delete_all_conversations():
    try:
        conn = sqlite3.connect("companion.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()
        return {"status": "all deleted"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: int):
    try:
        conn = sqlite3.connect("companion.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        conn.close()
        return JSONResponse(content={"status": "deleted", "id": conv_id})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/crud", response_class=HTMLResponse)
async def crud_ui(request: Request):
    return templates.TemplateResponse("crud.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=lambda: asyncio.run(loop.run_forever()), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
