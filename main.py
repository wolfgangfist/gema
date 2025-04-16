import asyncio
import os
import platform
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
            # Don't queue too many requests - limit to last 3 to avoid memory issues
            if len(pending_user_inputs) >= 3:
                # Keep only the most recent requests
                pending_user_inputs = pending_user_inputs[-2:]
            pending_user_inputs.append((user_text, session_id))
            logger.info(f"Added user input to pending queue (total: {len(pending_user_inputs)}): '{user_text}'")
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
       
def audio_generation_thread(text, output_file):
    """
    Ultra-low overhead audio generation thread.
    Optimized for maximum speed by eliminating all unnecessary processing.
    """
    global is_speaking, interrupt_flag, audio_queue
    if not audio_gen_lock.acquire(blocking=False):
        logger.warning("Another audio generation is in progress, skipping this one")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "error", "message": "Audio generation busy, skipping synthesis"}),
            loop
        )
        return
    
    try:
        # First, clear any existing interrupt flag
        interrupt_flag = False
        
        # Reset model state completely
        if hasattr(generator, '_model') and hasattr(generator._model, 'reset_caches'):
            generator._model.reset_caches()
            # Also clear the text token cache if it exists
            if hasattr(generator, '_text_token_cache'):
                generator._text_token_cache = {}
        
        # Force CUDA synchronization and clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        is_speaking = True
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        all_audio_chunks = []
        
        # Minimal prebuffer for faster start
        prebuffer_chunks = []
        prebuffer_complete = False
        prebuffer_event = threading.Event()
        
        try:
            # Clear CUDA cache before generation
            torch.cuda.empty_cache()

            # Dynamically estimate the maximum audio length (in milliseconds)
            words = text.split()
            avg_wpm = 160
            words_per_second = avg_wpm / 60
            estimated_seconds = len(words) / words_per_second
            buffer_seconds = 0  # Extra buffer
            max_audio_length_ms = int((estimated_seconds + buffer_seconds) * 1000)

            # Start a separate thread to handle real-time inference
            def inference_worker():
                nonlocal prebuffer_complete, prebuffer_chunks

                # Lowercase the text to improve synthesis quality
                text_lower = text.lower()

                # Start a timer to measure generation latency
                generation_start = time.time()
                last_chunk_time = generation_start
                
                # Use smaller chunk sizes for more frequent updates
                chunk_counter = 0
                
                try:
                    for audio_chunk in generator.generate_stream(
                        text=text_lower,
                        speaker=config.voice_speaker_id,
                        context=reference_segments,
                        max_audio_length_ms=max_audio_length_ms,
                        temperature=0.8,
                        topk=50,
                    ):
                        # Check interrupt flag frequently and exit immediately if set
                        if interrupt_flag:
                            logger.info("Interrupt detected in generator - stopping generation")
                            # Break without additional processing to stop generation ASAP
                            break

                        # Track timing for analytics
                        now = time.time()
                        chunk_latency = now - last_chunk_time
                        if chunk_counter % 10 == 0:  # Log every 10 chunks
                            logger.debug(f"Audio chunk {chunk_counter} generated in {chunk_latency*1000:.1f}ms")
                        last_chunk_time = now
                        chunk_counter += 1
                        
                        # Process the audio chunk - ensure it's a valid numpy array
                        try:
                            # Check interrupt flag again before processing chunk
                            if interrupt_flag:
                                break
                                
                            chunk_array = audio_chunk.cpu().numpy().astype(np.float32)
                            if chunk_array.size == 0:
                                logger.warning("Received empty audio chunk from generator")
                                continue
                                
                            all_audio_chunks.append(audio_chunk)
                            
                            # Initial prebuffering - just collect one chunk for minimal delay
                            if not prebuffer_complete:
                                prebuffer_chunks.append(chunk_array)
                                # Use minimal prebuffer (just 1 chunk) for fastest start
                                prebuffer_complete = True
                                prebuffer_event.set()
                            else:
                                # Check interrupt flag again before queueing audio
                                if interrupt_flag:
                                    break
                                    
                                # Enqueue the audio chunk for playback
                                try:
                                    audio_queue.put(chunk_array)
                                except:
                                    logger.error("Failed to put audio chunk in queue")
                            
                            # Send chunk to WebSocket clients
                            try:
                                send_to_all_clients({
                                    "type": "audio_chunk",
                                    "audio": chunk_array.tolist(),
                                    "sample_rate": generator.sample_rate
                                })
                            except Exception as e:
                                logger.error(f"Error sending to clients: {e}")
                                
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}")
                    
                    # Log total generation time
                    total_time = time.time() - generation_start
                    if all_audio_chunks:
                        total_audio_seconds = sum(len(chunk) for chunk in all_audio_chunks) / generator.sample_rate
                        rtf = total_time / total_audio_seconds
                        logger.info(f"Audio generated in {total_time:.2f}s, RTF: {rtf:.2f}x")
                    
                    # Ensure prebuffer is released if generation finished before prebuffering
                    if not prebuffer_complete:
                        prebuffer_complete = True
                        prebuffer_event.set()
                    
                except Exception as e:
                    logger.error(f"Error in inference worker: {e}")
                    prebuffer_event.set()  # Ensure the event is set to prevent deadlock
            
            # Notify clients that audio generation is beginning
            try:
                send_to_all_clients({"type": "audio_status", "status": "generating"})
            except Exception as e:
                logger.error(f"Error sending generating status: {e}")
            
            # Start the inference thread
            inference_thread = threading.Thread(target=inference_worker, daemon=True)
            inference_thread.start()
            
            # Wait for initial prebuffer to fill (or fail)
            prebuffer_event.wait(timeout=1.0)  # Wait up to 1 second for prebuffering
            
            # Check for interrupt before proceeding
            if interrupt_flag:
                logger.info("Interrupt detected before playback started - aborting")
                # Stop inference thread 
                if inference_thread.is_alive():
                    # We can't join directly as it may be blocked in model
                    logger.info("Waiting for inference thread to stop")
                # Skip adding to audio queue
                raise InterruptedError("Generation interrupted")
            
            # Safety check prebuffered chunks
            valid_prebuffer_chunks = [chunk for chunk in prebuffer_chunks if isinstance(chunk, np.ndarray) and chunk.size > 0]
            
            # Flush prebuffer to playback queue with no additional processing
            if valid_prebuffer_chunks and not interrupt_flag:
                logger.info(f"Starting playback with {len(valid_prebuffer_chunks)} prebuffered chunks")
                for chunk in valid_prebuffer_chunks:
                    # Check for interrupt again
                    if interrupt_flag:
                        break
                    try:
                        audio_queue.put(chunk)
                    except:
                        logger.error("Failed to put prebuffer chunk in queue")
            
            # Wait for inference to complete
            inference_thread.join(timeout=30.0)  # Add timeout to prevent hanging

            if all_audio_chunks and not interrupt_flag:
                try:
                    complete_audio = torch.cat(all_audio_chunks)
                    save_audio_and_trim(output_file, "default", config.voice_speaker_id, complete_audio, generator.sample_rate)
                    add_segment(text.lower(), config.voice_speaker_id, complete_audio)
                except Exception as e:
                    logger.error(f"Error saving complete audio: {e}")

        except RuntimeError as e:
            logger.error(f"CUDA memory error during audio generation: {e}")
            # Force a more extensive CUDA cleanup
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if hasattr(generator, '_model'):
                        generator._model.to('cpu')
                        torch.cuda.empty_cache()
                        generator._model.to('cuda')
                except Exception as cuda_e:
                    logger.error(f"Error during CUDA cleanup: {cuda_e}")
                    
            try:
                audio_queue.put(None)
            except:
                pass
                
            try:
                asyncio.run_coroutine_threadsafe(
                    message_queue.put({"type": "error", "message": "Audio generation failed due to GPU memory error"}),
                    loop
                )
            except Exception as e:
                logger.error(f"Failed to send error message: {e}")
        except InterruptedError:
            logger.info("Audio generation was interrupted")
        except Exception as e:
            logger.error(f"Unexpected error in audio generation: {e}")
        finally:
            is_speaking = False
            # Signal end of generation to the playback thread
            try:
                audio_queue.put(None)
            except:
                pass
                
            try:
                asyncio.run_coroutine_threadsafe(
                    message_queue.put({"type": "audio_status", "status": "complete"}),
                    loop
                )
            except Exception as e:
                logger.error(f"Failed to send complete status: {e}")
                
            # Make sure CUDA is clean before processing next request
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # Process any pending user inputs
            try:
                with user_input_lock:
                    if pending_user_inputs:
                        user_text, session_id = pending_user_inputs.pop(0)
                        pending_user_inputs.clear()  # Clear any other pending inputs
                        # Add small delay to ensure complete cleanup
                        time.sleep(0.2)
                        threading.Thread(target=lambda: process_user_input(user_text, session_id), daemon=True).start()
            except Exception as e:
                logger.error(f"Failed to process pending inputs: {e}")
    finally:
        audio_gen_lock.release()

def audio_playback_thread():
    """
    Ultra-low overhead audio playback thread with minimal processing.
    Optimized for maximum speed by eliminating resampling and overlap processing.
    """
    global audio_queue, interrupt_flag
    # Initialize shared buffer and associated lock for thread-safe access by the callback
    shared_buffer = np.array([], dtype=np.float32)
    buffer_lock = threading.Lock()
    
    # Set higher process priority for better real-time performance
    try:
        import psutil
        process = psutil.Process()
        if platform.system() == 'Windows':
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            # Use higher priority for Linux
            process.nice(-1)  # -20 to 19, lower is higher priority
    except (ImportError, PermissionError, psutil.AccessDenied):
        logger.warning("Could not set process priority - continuing with default priority")
    
    # CSM sample rate used by the generator
    csm_sample_rate = 24000
    logger.info(f"Using CSM sample rate for playback: {csm_sample_rate}Hz")
    
    # Initialize state flags and output sample rate
    audio_device_available = False
    output_sample_rate = csm_sample_rate

    # Check for available audio output device
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_output = sd.query_devices(kind='output')
        logger.info(f"Default output device: {default_output['name']}")
        supported_rate = default_output.get('default_samplerate', csm_sample_rate)
        logger.info(f"Default device sample rate: {supported_rate}Hz")
        
        # Force using CSM sample rate directly to avoid resampling latency
        logger.info(f"Forcing direct playback at {csm_sample_rate}Hz for maximum speed")
        audio_device_available = True

    except Exception as e:
        logger.warning(f"Audio device initialization failed: {e}")
        logger.info("Running in headless mode - will save audio files but not play audio")
        audio_device_available = False

    # Counter and directory for fallback file saving
    file_counter = 0
    os.makedirs("audio/fallback", exist_ok=True)

    # Ultra-low latency buffer size
    buffer_size = 16 
    
    if audio_device_available:
        try:
            import sounddevice as sd
            # Define non-blocking callback to continuously deliver audio to the output device
            def callback(outdata, frames, time_info, status):
                nonlocal shared_buffer
                if status:
                    logger.warning(f"Audio callback status: {status}")
                
                with buffer_lock:
                    if len(shared_buffer) >= frames:
                        # Fill the output buffer with available samples
                        outdata[:, 0] = shared_buffer[:frames]
                        shared_buffer = shared_buffer[frames:]
                    else:
                        # Not enough samples: fill what's available and pad the rest with zeros
                        available = len(shared_buffer)
                        if available > 0:
                            outdata[:available, 0] = shared_buffer
                        outdata[available:, 0] = 0
                        shared_buffer = np.zeros(0, dtype=np.float32)  # Safety: use zeros instead of empty
                        
                        # Request more audio data to prevent starvation
                        if available < frames / 2:
                            # Less risky way to request more data
                            try:
                                audio_queue.put_nowait("request_more")
                            except queue.Full:
                                pass  # Ignore if queue is full
            
            # Use the CSM sample rate directly for the output stream
            stream = sd.OutputStream(
                samplerate=csm_sample_rate,  # Use the CSM sample rate directly
                channels=1,
                callback=callback,
                blocksize=buffer_size,
                latency=0.01  # Ultra-low latency setting
            )
            stream.start()
            logger.info(f"Started direct audio playback stream at {csm_sample_rate}Hz with blocksize {buffer_size}.")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            audio_device_available = False

    # In fallback mode, accumulate audio into a buffer for file writing
    fallback_buffer = np.zeros(0, dtype=np.float32)  # Safety: use zeros instead of empty

    # Main loop: retrieve audio chunks from the queue and process them
    while True:
        try:
            # Reduced timeout for more frequent checks
            chunk = audio_queue.get(timeout=0.01)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error getting item from audio queue: {e}")
            continue

        if chunk is None:
            # End of current audio generation signal
            logger.debug("Received end-of-generation signal in playback thread")
            # Clear buffer on stop command to prevent playing stale audio
            with buffer_lock:
                shared_buffer = np.zeros(0, dtype=np.float32)
            break
            
        if isinstance(chunk, str) and chunk == "request_more":
            # This is just a signal that the buffer is low, not actual audio data
            continue

        # Make sure we're dealing with a valid chunk
        if not isinstance(chunk, np.ndarray) or chunk.size == 0:
            continue

        # Check if interrupted - if so, clear buffer and exit
        if interrupt_flag:
            logger.info("Interrupt detected in playback thread - clearing buffer")
            with buffer_lock:
                shared_buffer = np.zeros(0, dtype=np.float32)
            break

        # Process the audio chunk
        try:
            if audio_device_available:
                # Direct path with minimal processing - just add to buffer
                with buffer_lock:
                    if shared_buffer.size > 0:
                        shared_buffer = np.concatenate((shared_buffer, chunk))
                    else:
                        shared_buffer = chunk.copy()
            else:
                # In fallback mode: accumulate and then save to file when the buffer is long enough
                if fallback_buffer.size > 0:
                    fallback_buffer = np.concatenate((fallback_buffer, chunk))
                else:
                    fallback_buffer = chunk.copy()
                if len(fallback_buffer) >= csm_sample_rate:  # 1 second of audio
                    file_path = f"audio/fallback/chunk_{file_counter}.wav"
                    try:
                        import scipy.io.wavfile as wavfile
                        wavfile.write(file_path, csm_sample_rate, fallback_buffer)
                        logger.info(f"Saved audio chunk to {file_path}")
                        file_counter += 1
                    except Exception as e:
                        logger.error(f"Error saving audio file: {e}")
                    fallback_buffer = np.zeros(0, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    # End of generation: flush any remaining audio in the buffer
    if audio_device_available:
        try:
            # If interrupted, don't wait - clear buffer immediately
            if interrupt_flag:
                with buffer_lock:
                    shared_buffer = np.zeros(0, dtype=np.float32)
            else:
                # Otherwise, give the stream a short period to play out remaining audio
                timeout_time = time.time() + 0.2
                while shared_buffer.size > 0 and time.time() < timeout_time:
                    time.sleep(0.01)
            
            # Don't close stream - just clear buffer and let it keep running for next generation
            logger.info("Audio playback for current generation completed.")
        except Exception as e:
            logger.error(f"Error finishing audio playback: {e}")
    else:
        if fallback_buffer.size >= csm_sample_rate / 4:
            file_path = f"audio/fallback/chunk_{file_counter}.wav"
            try:
                import scipy.io.wavfile as wavfile
                wavfile.write(file_path, csm_sample_rate, fallback_buffer)
                logger.info(f"Saved final audio chunk to {file_path}")
            except Exception as e:
                logger.error(f"Error saving final audio file: {e}")
    
    # Start a new playback thread for the next generation
    threading.Thread(target=audio_playback_thread, daemon=True, name="audio_playback").start()
    
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
        
        # Set interrupt flag first before any other operations
        interrupt_flag = True
        
        # Clear the audio queue to stop additional audio from being processed
        try:
            # Put None at the front of the queue to signal stop
            temp_queue = queue.Queue()
            temp_queue.put(None)
            
            # Drain the existing queue to clear any pending chunks
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
                    
            # Now put the None signal
            audio_queue.put(None)
        except:
            # If queue operations fail, at least we have the interrupt flag set
            pass
        
        # Force CUDA synchronization to help prevent errors in future operations
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except:
                pass
        
        # Reset VAD to prepare for new input
        vad_processor.reset()
        
        # Notify client of interruption
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "audio_status", "status": "interrupted"}),
            loop
        )
        return True
    return False

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