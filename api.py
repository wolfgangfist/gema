from generator import load_csm_1b

import torchaudio
import torch

import os
import io

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader

from pydantic import BaseModel

# --------------- Init FastAPI app ---------------
app = FastAPI()

# --------------- Check CUDA ---------------
if not torch.cuda.is_available():
    print("❌ CUDA is not available. This API requires a GPU.")
    raise RuntimeError("CUDA is not available. This API requires a GPU.")

# --------------- Load model ---------------
print("🚀 Loading CSM model onto GPU...")
generator = load_csm_1b(device="cuda")
print("✅ Model loaded.")

# Make sure voices folder exists
os.makedirs("voices", exist_ok=True)

# --------------- Api key ---------------
API_KEY = os.getenv("CSM_API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

if API_KEY == "your-secret-key":
    print("❌ No CSM_API_KEY environment variable set. Using default insecure key.")
else:
    print("✅ CSM_API_KEY is set from environment.")

def verify_api_key(auth_header: str = Depends(api_key_header)):
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    token = auth_header.split("Bearer ")[-1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.on_event("startup")
async def startup_event():
    print("✅ CSM API is live and listening on http://0.0.0.0:5537/speak")

@app.get("/")
def root():
    return {"status": "✅ CSM API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


# --------------- Define speak request format ---------------
class SpeechRequest(BaseModel):
    text: str = "Hello Josh"
    speaker: int = 0
    max_audio_length: int = 10000

@app.post("/speak")
def generate_speech(req: SpeechRequest, _: str = Depends(verify_api_key)):
    print(f"🎙️ Generating speech for: '{req.text}' with speaker {req.speaker}")
    try:
        # Generate audio
        audio = generator.generate(
            text=req.text,
            speaker=req.speaker,
            context=[],
            max_audio_length_ms=req.max_audio_length,
        )

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
        buffer.seek(0)

        print(f"✅ Audio generated successfully")

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "Content-Disposition": "inline; filename=output.wav",
                "Cache-Control": "no-cache",
                "Content-Length": str(buffer.getbuffer().nbytes)
            }
        )

    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))