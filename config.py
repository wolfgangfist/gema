# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Server Configuration ---
# Fetch from environment variables with defaults
DEVICE_OVERRIDE = os.getenv("DEVICE") # Optional override (e.g., "cpu", "cuda")
RESPONSE_CHUNK_SIZE_MS = int(os.getenv("RESPONSE_CHUNK_SIZE_MS", 200))
MAX_CONTEXT_SEGMENTS = int(os.getenv("MAX_CONTEXT_SEGMENTS", 8))
USER_SPEAKER_ID = int(os.getenv("USER_SPEAKER_ID", 0))
AI_SPEAKER_ID = int(os.getenv("AI_SPEAKER_ID", 1))
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "sesame/csm-1b")
PROMPT_A_FILENAME = os.getenv("PROMPT_A_FILENAME", "prompts/conversational_a.wav")
PROMPT_B_FILENAME = os.getenv("PROMPT_B_FILENAME", "prompts/conversational_b.wav")
# You might want to load prompt text from env or files too
PROMPT_A_TEXT = os.getenv("PROMPT_A_TEXT", "like revising for an exam I'd have to try and like keep up the momentum because I'd start really early")
PROMPT_B_TEXT = os.getenv("PROMPT_B_TEXT", "like a super Mario level. Like it's very like high detail. And like, once you get into the park")
GENERATION_MAX_MS = int(os.getenv("GENERATION_MAX_MS", 20000))
GENERATION_TEMP = float(os.getenv("GENERATION_TEMP", 0.9))

# VAD Configuration
VAD_SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", 16000))
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", 30))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", 3))
VAD_SILENCE_FRAMES_THRESHOLD = int(os.getenv("VAD_SILENCE_FRAMES_THRESHOLD", 15))

# --- Client Configuration ---
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "http://localhost:7860") # Server URL
HF_TOKEN = os.getenv("HF_TOKEN") # Optional HF Token if needed by client
INTERFACE_TITLE = os.getenv("INTERFACE_TITLE", "CSM Streaming Interface")
INTERFACE_DESCRIPTION = os.getenv("INTERFACE_DESCRIPTION", "Speak to interact with the CSM model.")
AUTOPLAY_RESPONSES = os.getenv("AUTOPLAY_RESPONSES", "True").lower() == "true"
SHARE_GRADIO = os.getenv("SHARE_GRADIO", "False").lower() == "true" # Gradio sharing

# --- Shared Configuration ---
# Calculate VAD bytes per frame based on other VAD settings
VAD_BYTES_PER_FRAME = (VAD_SAMPLE_RATE * VAD_FRAME_MS // 1000) * 2 # 16-bit PCM 