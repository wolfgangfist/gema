import gradio as gr
import numpy as np
import soundfile as sf
import base64
import io
import asyncio
import websockets
import json
import logging
import traceback
import time
import os
import sys # Added
import uuid # Added for session ID
from queue import Queue, Empty
import threading
from enum import Enum # Added for message types

# Adjust path to import config from root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.path.dirname(SCRIPT_DIR) # Assuming client is one level down
ROOT_DIR = os.path.dirname(CWD)
sys.path.append(ROOT_DIR)

# --- Configuration Loading (Now from root config) ---
try:
    import config # Import from root config.py
    # Derive WebSocket URL from configured HTTP URL
    ws_proto = "wss://" if config.ENDPOINT_URL.startswith("https://") else "ws://"
    ws_host = config.ENDPOINT_URL.replace("https://", "").replace("http://", "")
    # Use UUID for session ID
    SESSION_ID = f"gradio_client_{uuid.uuid4()}"
    WEBSOCKET_URL = f"{ws_proto}{ws_host}/ws/{SESSION_ID}"
    logging.info(f"Root configuration loaded. WebSocket URL: {WEBSOCKET_URL}")
    # Use other config values directly: config.INTERFACE_TITLE, config.INTERFACE_DESCRIPTION, etc.
except ImportError:
    logging.error("Root config.py not found or cannot be imported. Using fallback defaults.")
    WEBSOCKET_URL = "ws://localhost:7860/ws/test_session"
    # Set fallback defaults for required client configs if root config fails
    config = type('obj', (object,), {
        'INTERFACE_TITLE': "CSM Streaming Interface (Default Config)",
        'INTERFACE_DESCRIPTION': "Default Description - Failed to load config",
        'AUTOPLAY_RESPONSES': True,
        'SHARE_GRADIO': False
    })()
except Exception as e:
    logging.error(f"Error loading or processing configuration: {e}")
    # Handle other potential config loading errors, provide defaults
    # ... (provide necessary defaults as above)

# Message Types Enum (Define locally or import if shared)
class MSG_TYPE(str, Enum):
    START = "start"
    AUDIO_CHUNK = "audio_chunk"
    STOP = "stop"
    RESET = "reset"
    # Server -> Client types
    STATUS = "status"
    AUDIO_CHUNK_RESPONSE = "audio_chunk"
    RESPONSE_END = "response_end"
    # Status Values
    INFO = "info"
    ERROR = "error"
    WARNING = "warning"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streaming_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting Streaming Gradio Interface Script")

# --- Global State & Communication Queues ---
# Use thread-safe queues for communication between Gradio callbacks and async WebSocket tasks
websocket_connection = None
receive_queue = Queue() # Queue for messages received from server
send_queue = Queue()    # Queue for messages/chunks to send to server
listener_task = None
sender_task = None
stop_event = threading.Event() # Signal to stop async tasks

# --- WebSocket Coroutines --- (Use MSG_TYPE enum)

async def websocket_listener(uri):
    """Listens for messages from the WebSocket server and puts them in receive_queue."""
    global websocket_connection
    logger.info(f"Listener: Attempting to connect to {uri}")
    try:
        async with websockets.connect(uri) as websocket:
            websocket_connection = websocket # Store active connection
            logger.info(f"Listener: WebSocket connected to {uri}")
            receive_queue.put({"type": MSG_TYPE.STATUS.value, "status": MSG_TYPE.INFO.value, "message": "Connected to server."}) # Signal connection success
            while not stop_event.is_set():
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0) # Timeout to allow checking stop_event
                    try:
                        data = json.loads(message)
                        receive_queue.put(data)
                        # logger.debug(f"Listener: Received message: {data.get('type')}")
                    except json.JSONDecodeError:
                        logger.warning(f"Listener: Received non-JSON message: {message[:100]}")
                except asyncio.TimeoutError:
                    continue # No message received, check stop_event again
                except websockets.exceptions.ConnectionClosedOK:
                    logger.info("Listener: WebSocket connection closed normally.")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(f"Listener: WebSocket connection closed with error: {e}")
                    receive_queue.put({"type": MSG_TYPE.STATUS.value, "status": MSG_TYPE.ERROR.value, "message": f"Connection closed error: {e}"})
                    break
    except Exception as e:
        logger.error(f"Listener: WebSocket connection failed: {e}", exc_info=True)
        receive_queue.put({"type": MSG_TYPE.STATUS.value, "status": MSG_TYPE.ERROR.value, "message": f"Connection failed: {e}"})
    finally:
        logger.info("Listener: Stopped.")
        websocket_connection = None
        stop_event.set() # Ensure sender also stops if listener fails/stops
        # Put a final signal? Maybe not needed if UI handles disconnect state

async def websocket_sender(uri):
    """Takes messages from send_queue and sends them to the WebSocket server."""
    # Need connection from listener
    while websocket_connection is None and not stop_event.is_set():
        await asyncio.sleep(0.1) # Wait for listener to establish connection

    if websocket_connection and not stop_event.is_set():
        logger.info("Sender: Ready.")
        while not stop_event.is_set():
            try:
                message_to_send = send_queue.get_nowait() # Non-blocking get
                try:
                    await websocket_connection.send(json.dumps(message_to_send))
                    # logger.debug(f"Sender: Sent message: {message_to_send.get('type')}")
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Sender: Connection closed, cannot send message.")
                    stop_event.set() # Stop sender if connection is closed
                    break
                except Exception as e:
                    logger.error(f"Sender: Error sending message: {e}")
                    # Optionally put back in queue or signal error
            except Empty:
                await asyncio.sleep(0.01) # Wait if queue is empty
    logger.info("Sender: Stopped.")

# --- Thread Functions to Run Async Tasks --- (Gradio runs in threads)

def run_async_task(loop, coro):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)

def start_websocket_threads():
    """Starts the sender and listener tasks in separate threads."""
    global listener_task, sender_task, stop_event
    stop_event.clear()
    loop = asyncio.new_event_loop()
    listener_coro = websocket_listener(WEBSOCKET_URL)
    listener_task = threading.Thread(target=run_async_task, args=(loop, listener_coro), daemon=True)
    listener_task.start()
    logger.info("Listener thread started.")

    # Sender uses the same loop but waits for connection
    sender_coro = websocket_sender(WEBSOCKET_URL)
    sender_task = threading.Thread(target=run_async_task, args=(loop, sender_coro), daemon=True)
    sender_task.start()
    logger.info("Sender thread started.")

def stop_websocket_threads():
    """Signals the async tasks to stop."""
    global listener_task, sender_task
    stop_event.set()
    logger.info("Stop event set for async tasks.")
    # Optional: Join threads if needed, but daemon=True might be sufficient
    # if listener_task and listener_task.is_alive():
    #     listener_task.join(timeout=2)
    # if sender_task and sender_task.is_alive():
    #     sender_task.join(timeout=2)
    # logger.info("Async threads joined.")
    listener_task = None
    sender_task = None

# --- Gradio UI & Logic ---

def build_streaming_interface():
    with gr.Blocks(title=config.INTERFACE_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {config.INTERFACE_TITLE}")
        gr.Markdown(config.INTERFACE_DESCRIPTION)
        gr.Markdown("Click 'Start Conversation', speak, then click 'Stop Conversation'.")
        status_box = gr.Textbox(label="Status", value="Idle", interactive=False)

        # State variables
        is_active = gr.State(False)
        client_state = gr.State({
            "connection": "disconnected",
            "receiving_audio": False,
            "response_buffer": [],
            "response_sample_rate": 16000 # Default, update from server
        })

        with gr.Row():
            chatbot = gr.Chatbot(label="Conversation", height=450, show_label=False)

        with gr.Row():
            audio_input = gr.Audio(
                label="Your Speech",
                sources=["microphone"],
                streaming=True,
                type="numpy",
                elem_id="audio-input",
                interactive=False # Start disabled
            )
            audio_output = gr.Audio(
                label="AI Response",
                type="numpy",
                autoplay=config.AUTOPLAY_RESPONSES,
                interactive=False,
                elem_id="audio-output"
            )

        with gr.Row():
            start_btn = gr.Button("Start Conversation", variant="primary")
            stop_btn = gr.Button("Stop Conversation", visible=False)
            reset_btn = gr.Button("Reset Context")

        # --- Gradio Event Handlers --- #

        def handle_start_button(current_state):
            if current_state["connection"] == "disconnected":
                logger.info("UI: Start button clicked, starting WS threads.")
                start_websocket_threads()
                # UI updates happen based on messages from receive_queue
                new_state = current_state.copy()
                new_state["connection"] = "connecting"
                return {
                    is_active: True,
                    client_state: new_state,
                    start_btn: gr.Button(visible=False),
                    stop_btn: gr.Button(visible=True),
                    audio_input: gr.Audio(interactive=True),
                    status_box: "Connecting...",
                    chatbot: [] # Clear chatbot on new connection
                }
            else:
                logger.warning("UI: Start clicked but already connected/connecting.")
                return {status_box: "Already connected or connecting."}

        start_btn.click(
            fn=handle_start_button,
            inputs=[client_state],
            outputs=[is_active, client_state, start_btn, stop_btn, audio_input, status_box, chatbot]
        )

        def handle_stop_button(current_state):
            logger.info("UI: Stop button clicked.")
            send_queue.put({"type": MSG_TYPE.STOP.value}) # Signal stop to backend
            stop_websocket_threads() # Signal threads to stop
            # Clear response buffer on manual stop
            new_state = current_state.copy()
            new_state["connection"] = "disconnected"
            new_state["receiving_audio"] = False
            new_state["response_buffer"] = []

            return {
                is_active: False,
                client_state: new_state,
                start_btn: gr.Button(visible=True),
                stop_btn: gr.Button(visible=False),
                audio_input: gr.Audio(interactive=False, value=None),
                status_box: "Disconnected. Click Start to begin.",
                audio_output: gr.Audio(value=None) # Clear output
            }

        stop_btn.click(
            fn=handle_stop_button,
            inputs=[client_state],
            outputs=[is_active, client_state, start_btn, stop_btn, audio_input, status_box, audio_output]
        )

        def handle_reset_button(current_state):
            logger.info("UI: Reset button clicked.")
            if current_state["connection"] == "connected":
                 send_queue.put({"type": MSG_TYPE.RESET.value})
                 # Clear response buffer and history
                 new_state = current_state.copy()
                 new_state["response_buffer"] = []
                 return {
                     chatbot: [],
                     audio_output: gr.Audio(value=None),
                     client_state: new_state,
                     status_box: "Reset signal sent."
                 }
            else:
                return {status_box: "Not connected."} # Or maybe just clear local chatbot?

        reset_btn.click(
            fn=handle_reset_button,
            inputs=[client_state],
            outputs=[chatbot, audio_output, client_state, status_box]
        )

        def process_input_audio_chunk(audio_chunk, is_active_state):
            """Called by Gradio when a new audio chunk is available from the mic."""
            if not is_active_state or audio_chunk is None:
                return # Don't send if not active

            sample_rate, audio_array = audio_chunk

            # Need to run this in background? Maybe Gradio handles it.
            try:
                buffer = io.BytesIO()
                # Convert to 16-bit PCM WAV for potentially smaller size
                audio_array_int16 = (audio_array * 32767).astype(np.int16)
                sf.write(buffer, audio_array_int16, sample_rate, format='WAV', subtype='PCM_16')
                buffer.seek(0)
                chunk_bytes = buffer.read()
                chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')

                # Put message in send queue for async sender thread
                send_queue.put({
                    "type": MSG_TYPE.AUDIO_CHUNK.value,
                    "audio_base64": chunk_base64,
                    # "sample_rate": sample_rate # Server gets SR on start
                })
                # logger.debug(f"UI: Queued audio chunk (SR={sample_rate}, {len(chunk_bytes)} bytes).")
            except Exception as e:
                logger.error(f"UI: Error processing audio chunk: {e}")

            # This handler doesn't update output components directly
            return

        audio_input.stream(
            fn=process_input_audio_chunk,
            inputs=[audio_input, is_active],
            outputs=None, # Don't update UI directly from here
            every=0.2 # Send chunks every 200ms (adjust as needed)
        )

        # --- Background Processor for Server Messages --- #
        # This function runs periodically to check the receive_queue and update the UI
        def process_server_messages(current_history, current_state):
            new_status = None
            final_audio_output = gr.Audio() # No change by default
            new_history = current_history
            new_state = current_state

            try:
                while True: # Process all messages currently in queue
                    msg = receive_queue.get_nowait()
                    msg_type = msg.get("type")
                    # logger.debug(f"UI Processor: Got message type {msg_type}")

                    if msg_type == MSG_TYPE.STATUS.value:
                        new_status = msg.get("message", "Unknown status")
                        logger.info(f"UI Status: {new_status}")
                        if "Connected to server" in new_status:
                            new_state["connection"] = "connected"
                            # Send start message ONLY after connection confirmed
                            logger.info("UI: Sending start message to server.")
                            send_queue.put({
                                "type": MSG_TYPE.START.value,
                                "sample_rate": 44100 # TODO: Get actual SR from mic?
                            })

                        if "Connection failed" in new_status or "Connection closed" in new_status:
                             new_state["connection"] = "disconnected"
                             # Trigger UI update similar to stop button
                             # This part is tricky - need to update state that affects UI elements
                             # For simplicity, just update status for now

                    elif msg_type == MSG_TYPE.AUDIO_CHUNK_RESPONSE.value:
                        if new_state["connection"] == "connected":
                            new_state["receiving_audio"] = True
                            audio_b64 = msg.get("audio_base64")
                            sr = msg.get("sample_rate")
                            if audio_b64 and sr:
                                try:
                                    # logger.debug(f"UI: Decoding received audio chunk (SR={sr})...")
                                    audio_bytes = base64.b64decode(audio_b64)
                                    buffer = io.BytesIO(audio_bytes)
                                    audio_array, _ = sf.read(buffer)
                                    new_state["response_buffer"].append(audio_array.astype(np.float32))
                                    new_state["response_sample_rate"] = sr # Store SR
                                except Exception as e:
                                    logger.error(f"UI: Error decoding received chunk: {e}")
                            else:
                                 logger.warning("UI: Received audio chunk message missing data or sample rate.")

                    elif msg_type == MSG_TYPE.RESPONSE_END.value:
                        if new_state["response_buffer"]:
                            logger.info("UI: Received response end, combining buffer.")
                            final_np = np.concatenate(new_state["response_buffer"]).astype(np.float32)
                            sr = new_state["response_sample_rate"]
                            final_audio_output = gr.Audio(value=(sr, final_np))
                            new_history = current_history + [(None, f"(Audio Response: {len(final_np)/sr:.2f}s)")] # Add placeholder to chat
                            # Clear buffer after processing
                            new_state["response_buffer"] = []
                        else:
                             logger.info("UI: Received response end but buffer is empty.")
                        new_state["receiving_audio"] = False
                    # Add handling for other message types if needed

            except Empty:
                pass # No more messages in queue for now
            except Exception as e:
                logger.error(f"UI Processor: Error processing queue: {e}", exc_info=True)
                new_status = f"Client Error: {e}"

            # Prepare the updates dictionary
            updates = {}
            if new_status:
                updates[status_box] = new_status
            # Check if final_audio_output was updated (is not default gr.Audio())
            if final_audio_output.value is not None:
                updates[audio_output] = final_audio_output
            if new_history != current_history:
                 updates[chatbot] = new_history
            # Always return the latest state
            updates[client_state] = new_state

            # Only return components that actually changed
            # This seems complex with Gradio's update mechanism. Return all potentially changed outputs.
            return new_status, new_history, new_state, final_audio_output.value

        # Use .load() to run the processor periodically
        demo.load(
            fn=process_server_messages,
            inputs=[chatbot, client_state],
            outputs=[status_box, chatbot, client_state, audio_output],
            every=0.1 # Check for messages every 100ms
        )

    return demo

# --- Launch --- #
if __name__ == "__main__":
    logger.info("Building and launching Streaming Gradio Interface")
    streaming_demo = build_streaming_interface()
    try:
        # queue needs to be passed if UI elements directly interact with it
        # but here we use demo.load as a poller
        streaming_demo.launch(share=config.SHARE_GRADIO)
        logger.info("Streaming Gradio Interface stopped")
    except Exception as e:
        logger.critical(f"Failed to launch Gradio demo: {e}", exc_info=True)
    finally:
        # Ensure threads are stopped on exit
        stop_websocket_threads() 