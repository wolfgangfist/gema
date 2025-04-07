import os
import base64
import io
import soundfile as sf
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import traceback
import asyncio
import json
import time
import webrtcvad
from enum import Enum
import sys # To adjust path for importing root config

# Adjust path to import config from root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR) # Assuming server is one level down from root
sys.path.append(ROOT_DIR)

try:
    import config # Import from root config.py
except ImportError as e:
    print(f"FATAL: Could not import root config.py. Ensure it exists and ROOT_DIR is correct: {e}")
    sys.exit(1)

# Assume core modules (generator, models, hf_hub) are in csm/core/ or installed
CORE_DIR = os.path.join(ROOT_DIR, 'csm', 'core')
sys.path.append(CORE_DIR) # Add core dir to path temporarily

try:
    # Attempt to import from core directory structure or installed package
    from core.generator import load_csm_1b, Segment, Generator
    from core.models import Model # If needed directly
    from huggingface_hub import hf_hub_download # This is likely installed, not in core
except ImportError as e:
    print(f"Error importing core modules (generator/models): {e}. Ensure they are in csm/core/ or installed correctly.")
    # Decide how critical this is - maybe server can start but warn?
    # For now, let it potentially fail later if generator is None.

# Reset path if needed, though usually append is fine for session
# sys.path.remove(ROOT_DIR)
# sys.path.remove(CORE_DIR)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
if config.DEVICE_OVERRIDE:
    DEVICE = config.DEVICE_OVERRIDE
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
     # Check for MPS support more robustly
     DEVICE = "mps"
else:
    DEVICE = "cpu"

os.environ["NO_TORCH_COMPILE"] = "1"
RESPONSE_CHUNK_SIZE_MS = config.RESPONSE_CHUNK_SIZE_MS # Send audio back in chunks of this duration
MAX_CONTEXT_SEGMENTS = config.MAX_CONTEXT_SEGMENTS # Max conversation segments (user+ai) to keep, excluding initial prompts
USER_SPEAKER_ID = config.USER_SPEAKER_ID
AI_SPEAKER_ID = config.AI_SPEAKER_ID

# VAD Configuration
VAD_SAMPLE_RATE = config.VAD_SAMPLE_RATE  # webrtcvad supports 8000, 16000, 32000, 48000
VAD_FRAME_MS = config.VAD_FRAME_MS       # Frame duration in ms (10, 20, or 30)
VAD_BYTES_PER_FRAME = (VAD_SAMPLE_RATE * VAD_FRAME_MS // 1000) * 2 # 16-bit PCM
VAD_AGGRESSIVENESS = config.VAD_AGGRESSIVENESS  # 0 (least aggressive) to 3 (most aggressive)
VAD_SILENCE_FRAMES_THRESHOLD = config.VAD_SILENCE_FRAMES_THRESHOLD # Approx 15 * 30ms = 450ms of silence to trigger end-of-speech

# Message Types Enum
class MSG_TYPE(str, Enum):
    START = "start"
    AUDIO_CHUNK = "audio_chunk"
    STOP = "stop"
    RESET = "reset"
    # Server -> Client types
    STATUS = "status"
    AUDIO_CHUNK_RESPONSE = "audio_chunk" # Renamed for clarity
    RESPONSE_END = "response_end"
    # Status Values
    INFO = "info"
    ERROR = "error"
    WARNING = "warning"

# --- Global Variables ---
generator: Optional['Generator'] = None
speaker_prompts: List[Segment] = []

# --- State Management for WebSockets ---
class SessionState:
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        # Buffer for the Generator model (original sample rate, float32 numpy)
        self.audio_buffer: List[np.ndarray] = []
        self.accumulated_input_duration_s: float = 0.0
        self.sample_rate: Optional[int] = None # Store sample rate from client chunk

        # VAD related state
        self.vad: Optional[webrtcvad.Vad] = None
        self.vad_audio_buffer: bytearray = bytearray() # Buffer for VAD (must be VAD_SAMPLE_RATE, 16-bit PCM mono)
        self.is_speaking: bool = False
        self.silence_frames_count: int = 0

        # Processing state
        self.is_processing: bool = False
        self.context: List[Segment] = [] # Initialize context carefully later

        # Initialize VAD if possible
        try:
            self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
            logger.debug(f"[{self.session_id}] VAD Initialized.")
        except Exception as e:
             logger.error(f"[{self.session_id}] Failed to initialize VAD: {e}")
             self.vad = None # Ensure VAD is None if init fails

session_states: Dict[str, SessionState] = {}

# --- Helper Functions ---

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    """Loads and resamples prompt audio."""
    try:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0) # Ensure mono
        if sample_rate != target_sample_rate:
            logger.info(f"Resampling prompt {os.path.basename(audio_path)} from {sample_rate}Hz to {target_sample_rate}Hz")
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
            )
        logger.info(f"Loaded prompt: {os.path.basename(audio_path)}")
        return audio_tensor
    except Exception as e:
        logger.error(f"Error loading prompt audio {audio_path}: {e}")
        raise

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    """Creates a Segment object for a speaker prompt."""
    # Assuming DEVICE is set globally
    audio_tensor = load_prompt_audio(audio_path, sample_rate).to(DEVICE)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

# No need for decode_base64_audio here as we receive bytes/json from websocket

def encode_audio_base64(audio_tensor: torch.Tensor, sample_rate: int) -> str:
    """Encodes a torch Tensor to a base64 audio string (for sending back)."""
    buffer = io.BytesIO()
    # Ensure tensor is on CPU and correct format for soundfile
    audio_numpy = audio_tensor.squeeze().cpu().numpy()
    # Ensure it's float32 or int16, soundfile might require specific types
    if audio_numpy.dtype != np.int16:
        audio_numpy = (audio_numpy * 32767).astype(np.int16)

    sf.write(buffer, audio_numpy, sample_rate, format='WAV', subtype='PCM_16') # Specify subtype
    buffer.seek(0)
    audio_bytes = buffer.read()
    base64_data = base64.b64encode(audio_bytes).decode('utf-8')
    # No prefix needed if client expects raw base64 in JSON
    return base64_data

def ensure_vad_format(audio_bytes: bytes, original_sr: int) -> Optional[bytes]:
    """Converts audio bytes to the format required by VAD."""
    if original_sr == config.VAD_SAMPLE_RATE:
        return audio_bytes
    else:
        try:
            buffer = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(buffer, dtype='float32')
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != config.VAD_SAMPLE_RATE:
                 audio_tensor = torch.from_numpy(audio_array).float()
                 resampled_tensor = torchaudio.functional.resample(
                     audio_tensor, orig_freq=sr, new_freq=config.VAD_SAMPLE_RATE
                 )
                 audio_array = resampled_tensor.numpy()
            audio_int16 = (audio_array * 32767).astype(np.int16)
            return audio_int16.tobytes()
        except Exception as e:
            logger.error(f"Error converting audio for VAD (orig_sr={original_sr}): {e}")
            return None

async def _send_status_safe(websocket: WebSocket, session_id: str, status: MSG_TYPE, message: str):
    """Safely sends a status message, handling potential connection errors."""
    try:
        if websocket.client_state == websocket.client_state.CONNECTED:
             await websocket.send_json({
                 "type": MSG_TYPE.STATUS,
                 "status": status.value, # Use enum value
                 "message": message
             })
    except Exception as e:
        logger.warning(f"[{session_id}] Failed to send status message over websocket: {e}")

async def _process_vad_for_chunk(session_state: SessionState, vad_bytes: bytes, websocket: WebSocket) -> bool:
    """Processes VAD frames from bytes, returns True if inference should be triggered."""
    trigger_inference = False
    session_id = session_state.session_id # For logging

    if not session_state.vad:
        logger.warning(f"[{session_id}] VAD not initialized, cannot process VAD.")
        return False # Cannot trigger if VAD doesn't exist

    session_state.vad_audio_buffer.extend(vad_bytes)

    # Process VAD buffer in frames
    while len(session_state.vad_audio_buffer) >= config.VAD_BYTES_PER_FRAME:
        frame = session_state.vad_audio_buffer[:config.VAD_BYTES_PER_FRAME]
        session_state.vad_audio_buffer = session_state.vad_audio_buffer[config.VAD_BYTES_PER_FRAME:]

        try:
            is_speech = session_state.vad.is_speech(frame, config.VAD_SAMPLE_RATE)
            # logger.debug(f"VAD Frame: {'SPEECH' if is_speech else 'SILENCE'}")

            if is_speech:
                if not session_state.is_speaking:
                    logger.debug(f"[{session_id}] Speech start detected.")
                    # Optional: Send VAD status to client
                    # await _send_status_safe(websocket, session_id, MSG_TYPE.INFO, "VAD: Speaking")
                session_state.is_speaking = True
                session_state.silence_frames_count = 0
            elif session_state.is_speaking: # Speech was active, now silence
                session_state.silence_frames_count += 1
                if session_state.silence_frames_count >= config.VAD_SILENCE_FRAMES_THRESHOLD:
                    logger.info(f"[{session_id}] End of speech detected by VAD ({session_state.silence_frames_count * config.VAD_FRAME_MS}ms silence).")
                    session_state.is_speaking = False
                    session_state.silence_frames_count = 0 # Reset silence count
                    session_state.vad_audio_buffer = bytearray() # Clear VAD buffer on trigger

                    # Optional: Send VAD status to client
                    # await _send_status_safe(websocket, session_id, MSG_TYPE.INFO, "VAD: Silence detected")

                    # Trigger processing ONLY if not already processing
                    if not session_state.is_processing and session_state.audio_buffer:
                        logger.info(f"[{session_id}] Flagging utterance processing via VAD.")
                        await _send_status_safe(websocket, session_id, MSG_TYPE.INFO, "Processing utterance (VAD trigger)...")
                        trigger_inference = True
                        # Break loop once triggered, process_full_utterance will handle buffer
                        break
                    elif session_state.is_processing:
                         logger.warning(f"[{session_id}] VAD detected end of speech, but already processing.")
                    elif not session_state.audio_buffer:
                         logger.info(f"[{session_id}] VAD detected end of speech, but no audio buffered.")

        except Exception as vad_err:
            logger.error(f"[{session_id}] Error during VAD frame processing: {vad_err}", exc_info=True)
            # Reset VAD state on error?
            session_state.is_speaking = False
            session_state.silence_frames_count = 0
            # Potentially notify client of VAD error
            await _send_status_safe(websocket, session_id, MSG_TYPE.ERROR, f"VAD processing error: {vad_err}")
            # Don't trigger inference if VAD failed
            trigger_inference = False
            break # Stop processing VAD frames for this chunk on error

    return trigger_inference

async def process_full_utterance(session_state: SessionState, websocket: WebSocket):
    """Processes accumulated audio, runs model (non-blocking), streams response."""
    if session_state.is_processing:
        logger.warning(f"[{session_state.session_id}] Already processing, skipping redundant request.")
        return # Avoid concurrent processing for the same session

    if not session_state.audio_buffer:
        logger.warning(f"[{session_state.session_id}] No audio in buffer to process.")
        # Don't send error to client here, VAD might trigger this legitimately
        return

    session_state.is_processing = True # Set flag immediately

    if generator is None or not session_state.context: # Check context is initialized
         logger.error("Generator or context not initialized!")
         await _send_status_safe(websocket, session_state.session_id, MSG_TYPE.ERROR, "Model or context not ready.")
         session_state.is_processing = False
         return

    if session_state.sample_rate is None:
         logger.error(f"[{session_state.session_id}] Sample rate not determined from client.")
         await _send_status_safe(websocket, session_state.session_id, MSG_TYPE.ERROR, "Sample rate unknown.")
         session_state.is_processing = False
         return

    # Make copies of data needed for the background task
    audio_buffer_copy = list(session_state.audio_buffer)
    context_copy = list(session_state.context) # Crucial for context safety
    input_sample_rate = session_state.sample_rate
    session_id = session_state.session_id # For logging inside task

    # Clear buffer *before* starting background task
    session_state.audio_buffer = []
    session_state.accumulated_input_duration_s = 0.0
    logger.info(f"[{session_id}] Cleared audio buffer, starting background processing.")

    async def background_inference_task():
        try:
            # 1. Combine buffer into a single tensor
            full_audio_np = np.concatenate(audio_buffer_copy)
            full_audio_tensor = torch.from_numpy(full_audio_np).float()

            # 2. Resample if client SR differs from model SR
            if input_sample_rate != generator.sample_rate:
                logger.info(f"[{session_id}] Resampling input from {input_sample_rate}Hz to {generator.sample_rate}Hz")
                full_audio_tensor = torchaudio.functional.resample(
                    full_audio_tensor, orig_freq=input_sample_rate, new_freq=generator.sample_rate
                )

            full_audio_tensor = full_audio_tensor.to(DEVICE)
            logger.info(f"[{session_id}] Combined & Resampled audio tensor shape: {full_audio_tensor.shape}")

            # 3. Create user Segment and update context (use constants)
            input_segment = Segment(text="", speaker=config.USER_SPEAKER_ID, audio=full_audio_tensor)
            current_context = context_copy + [input_segment]

            # 4. Run model generation in a separate thread (use constants)
            logger.info(f"[{session_id}] Running generator in background thread...")
            start_time = time.time()
            response_audio_tensor = await asyncio.to_thread(
                generator.generate,
                text="",
                speaker=config.AI_SPEAKER_ID, # Use constant
                context=current_context,
                max_audio_length_ms=config.GENERATION_MAX_MS,
                temperature=config.GENERATION_TEMP
            )
            end_time = time.time()
            logger.info(f"[{session_id}] Generation took {end_time - start_time:.2f}s")

            # 5. Update *session* context (use constants and moved MAX_CONTEXT_SEGMENTS)
            if session_id in session_states:
                 response_segment = Segment(text="", speaker=config.AI_SPEAKER_ID, audio=response_audio_tensor)
                 session_states[session_id].context.append(input_segment)
                 session_states[session_id].context.append(response_segment)
                 # Trim context using global constant
                 if len(session_states[session_id].context) > config.MAX_CONTEXT_SEGMENTS + len(speaker_prompts): # Adjust check
                     keep_segments = session_states[session_id].context[-(config.MAX_CONTEXT_SEGMENTS):] # Keep last N segments
                     initial_prompts = list(speaker_prompts) # Use a fresh copy
                     session_states[session_id].context = initial_prompts + keep_segments
                     logger.debug(f"[{session_id}] Session context trimmed to {len(session_states[session_id].context)} segments.")

            # 6. Stream response tensor back to client (use constants)
            logger.info(f"[{session_id}] Streaming response audio...")
            samples_per_chunk = int(generator.sample_rate * (config.RESPONSE_CHUNK_SIZE_MS / 1000.0))
            total_samples = response_audio_tensor.shape[0]

            if websocket.client_state == websocket.client_state.CONNECTED:
                 for i in range(0, total_samples, samples_per_chunk):
                     chunk_tensor = response_audio_tensor[i : i + samples_per_chunk]
                     chunk_base64 = encode_audio_base64(chunk_tensor, generator.sample_rate)
                     try:
                         await websocket.send_json({
                             "type": MSG_TYPE.AUDIO_CHUNK_RESPONSE.value, # Use enum value
                             "audio_base64": chunk_base64,
                             "sample_rate": generator.sample_rate
                         })
                         await asyncio.sleep(0.01)
                     except Exception as send_err:
                          logger.warning(f"[{session_id}] Failed to send audio chunk over websocket: {send_err}")
                          break

                 if websocket.client_state == websocket.client_state.CONNECTED:
                     await websocket.send_json({"type": MSG_TYPE.RESPONSE_END.value}) # Use enum value
                     logger.info(f"[{session_id}] Finished streaming response.")
            else:
                logger.warning(f"[{session_id}] Websocket disconnected before streaming could complete.")

        except Exception as e:
            logger.error(f"[{session_id}] Error during background processing: {e}", exc_info=True)
            # Use safe send helper
            await _send_status_safe(websocket, session_id, MSG_TYPE.ERROR, f"Processing error: {str(e)}")
        finally:
             if session_id in session_states:
                 session_states[session_id].is_processing = False
                 logger.info(f"[{session_id}] Background processing finished, reset processing flag.")

    # Launch the background task
    asyncio.create_task(background_inference_task())

# --- FastAPI Application ---
app = FastAPI(title="CSM Streaming Inference Endpoint", version="1.1")

@app.on_event("startup")
async def startup_event():
    """Load model and prompts when the server starts."""
    global generator, speaker_prompts
    logger.info(f"Starting up API on device: {DEVICE}")
    try:
        logger.info("Loading CSM-1B model...")
        generator = load_csm_1b(device=DEVICE)
        logger.info("Model loaded successfully.")
        logger.info(f"Model sample rate: {generator.sample_rate}Hz")

        logger.info("Downloading and loading speaker prompts...")
        prompt_a_path = hf_hub_download(repo_id=config.MODEL_REPO_ID, filename=config.PROMPT_A_FILENAME)
        prompt_b_path = hf_hub_download(repo_id=config.MODEL_REPO_ID, filename=config.PROMPT_B_FILENAME)
        prompt_a = prepare_prompt(config.PROMPT_A_TEXT, 0, prompt_a_path, generator.sample_rate)
        prompt_b = prepare_prompt(config.PROMPT_B_TEXT, 1, prompt_b_path, generator.sample_rate)
        speaker_prompts = [prompt_a, prompt_b]
        logger.info("Speaker prompts loaded and assigned globally.")

    except Exception as e:
        logger.error(f"Critical error during startup: {e}", exc_info=True)
        generator = None # Ensure generator is None if loading fails
        raise RuntimeError(f"Failed to initialize model or prompts: {e}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session: {session_id}")

    # Initialize state only if speaker_prompts are ready
    if not speaker_prompts:
        logger.error("Speaker prompts not loaded during startup. Cannot initialize session.")
        await websocket.close(code=1011, reason="Server not ready: Prompts failed to load.")
        return

    session_state = SessionState(session_id)
    if session_state.vad is None:
        logger.error(f"[{session_id}] VAD failed to initialize. Cannot process audio.")
        await websocket.close(code=1011, reason="Server VAD initialization failed.")
        return

    session_state.context = list(speaker_prompts) # Initialize context
    session_states[session_id] = session_state
    logger.info(f"[{session_id}] Initialized session state with {len(session_state.context)} context segments.")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Use Enum for safer access, handle potential KeyError
                msg_type_str = message.get("type")
                try:
                    msg_type = MSG_TYPE(msg_type_str) if msg_type_str else None
                except ValueError:
                     logger.warning(f"[{session_id}] Received unknown message type string: {msg_type_str}")
                     msg_type = None # Treat as unknown

                # logger.debug(f"[{session_id}] Received message: type={msg_type}")
            except json.JSONDecodeError:
                logger.warning(f"[{session_id}] Received non-JSON message: {data[:100]}")
                continue

            # Use Enum for comparison
            if session_state.is_processing and msg_type != MSG_TYPE.RESET:
                 logger.warning(f"[{session_id}] Received message type '{msg_type}' while processing, ignoring.")
                 continue

            if msg_type == MSG_TYPE.START:
                logger.info(f"[{session_id}] Received start signal (VAD active).")
                session_state.audio_buffer = []
                session_state.accumulated_input_duration_s = 0.0
                session_state.vad_audio_buffer = bytearray()
                session_state.is_speaking = False
                session_state.silence_frames_count = 0
                session_state.sample_rate = message.get("sample_rate")
                if session_state.sample_rate is None:
                     logger.warning(f"[{session_id}] Client did not send sample_rate with start message. Assuming {VAD_SAMPLE_RATE}Hz for VAD.")
                     session_state.sample_rate = generator.sample_rate if generator else 16000
                else:
                    logger.info(f"[{session_id}] Client sample rate set to: {session_state.sample_rate}Hz")

                await _send_status_safe(websocket, session_id, MSG_TYPE.INFO, "Session started, VAD active.")

            elif msg_type == MSG_TYPE.AUDIO_CHUNK:
                trigger_processing = False
                try:
                    audio_base64 = message.get("audio_base64")
                    if not audio_base64 or session_state.sample_rate is None:
                         logger.warning(f"[{session_id}] Ignoring audio chunk: No base64 data or sample rate unknown.")
                         continue
                    # VAD check moved to helper

                    # Decode audio
                    prefix, base64_data = f"data:audio/wav;base64,{audio_base64}".split(',', 1)
                    audio_bytes_original = base64.b64decode(base64_data)

                    # Buffer for generator (soundfile read)
                    buffer = io.BytesIO(audio_bytes_original)
                    audio_array, sr = sf.read(buffer, dtype='float32')

                    if sr != session_state.sample_rate:
                         logger.warning(f"[{session_id}] Chunk SR {sr} differs from session SR {session_state.sample_rate}. Ignoring chunk.")
                         continue

                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)

                    session_state.audio_buffer.append(audio_array.astype(np.float32))
                    chunk_duration_s = len(audio_array) / session_state.sample_rate
                    session_state.accumulated_input_duration_s += chunk_duration_s
                    # logger.debug(f"[{session_id}] Buffered audio chunk ({chunk_duration_s:.2f}s). Total: {session_state.accumulated_input_duration_s:.2f}s")

                    # Process VAD
                    vad_bytes = ensure_vad_format(audio_bytes_original, session_state.sample_rate)
                    if vad_bytes:
                        trigger_processing = await _process_vad_for_chunk(session_state, vad_bytes, websocket)
                    else:
                         logger.warning(f"[{session_id}] Could not convert audio chunk to VAD format.")
                         # Should we send an error here? Maybe just log for now.

                except Exception as e:
                     logger.error(f"[{session_id}] Error processing audio chunk: {e}", exc_info=True)
                     await _send_status_safe(websocket, session_id, MSG_TYPE.ERROR, f"Error processing audio chunk: {e}")
                     # Potentially reset VAD state on chunk error?
                     session_state.is_speaking = False
                     session_state.silence_frames_count = 0

                # Trigger inference outside the try/except block if flagged by VAD helper
                if trigger_processing:
                     await process_full_utterance(session_state, websocket)

            elif msg_type == MSG_TYPE.STOP:
                logger.info(f"[{session_id}] Received explicit stop signal.")
                if not session_state.is_processing:
                    if session_state.audio_buffer:
                        logger.info(f"[{session_id}] Triggering utterance processing via explicit stop.")
                        session_state.is_speaking = False
                        session_state.silence_frames_count = 0
                        session_state.vad_audio_buffer = bytearray()
                        await _send_status_safe(websocket, session_id, MSG_TYPE.INFO, "Processing utterance (stop trigger)...")
                        await process_full_utterance(session_state, websocket)
                    else:
                        logger.info(f"[{session_id}] Stop received but no audio buffered, doing nothing.")
                        await _send_status_safe(websocket, session_id, MSG_TYPE.INFO, "Stop received, no audio processed.")
                else:
                     logger.warning(f"[{session_id}] Received stop signal while already processing.")
                     await _send_status_safe(websocket, session_id, MSG_TYPE.WARNING, "Already processing previous utterance.")

            elif msg_type == MSG_TYPE.RESET:
                 logger.info(f"[{session_id}] Received reset signal.")
                 session_state.context = list(speaker_prompts)
                 session_state.audio_buffer = []
                 session_state.accumulated_input_duration_s = 0.0
                 session_state.vad_audio_buffer = bytearray()
                 session_state.is_speaking = False
                 session_state.silence_frames_count = 0
                 session_state.is_processing = False
                 logger.info(f"[{session_id}] Context and buffers reset.")
                 await _send_status_safe(websocket, session_id, MSG_TYPE.INFO, "Conversation context reset.")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"[{session_id}] Error in WebSocket handler: {e}", exc_info=True)
        # Use safe send helper
        await _send_status_safe(websocket, session_id, MSG_TYPE.ERROR, f"Server error: {str(e)}")
    finally:
        if session_id in session_states:
            del session_states[session_id]
            logger.info(f"Cleaned up state for session: {session_id}")

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    if generator is not None:
        return {"status": "ok", "message": "Model loaded."}
    else:
         raise HTTPException(status_code=503, detail="Service Unavailable: Model not ready.")

# --- Main execution (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local testing (including WebSocket)...")
    # Note: startup event is handled by uvicorn when run this way
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True) # Added reload for easier local dev 