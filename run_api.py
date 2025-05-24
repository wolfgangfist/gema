from generator import load_csm_1b

import torchaudio
import torch

import os
import io

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader


# --------------- Init FastAPI app ---------------
app = FastAPI()

# --------------- Check CUDA ---------------
if not torch.cuda.is_available():
    print("❌ CUDA is not available. This API requires a GPU.")
    raise RuntimeError("CUDA is not available. This API requires a GPU.")

# --------------- Load model ---------------
print("🚀 Loading CSM model onto GPU...")
#generator = load_csm_1b(device="cuda")
print("✅ Model loaded.")

INTERNAL_PORT = os.getenv("INTERNAL_PORT", 8888)

@app.on_event("startup")
async def startup_event():
    print(f"✅ CSM API is live and listening on http://0.0.0.0:{INTERNAL_PORT}/")

@app.get("/")
def root():
    return {"status": "✅ CSM API is running"}

# --------------- Import health router ---------------
from api.routes.health import health_router
app.include_router(health_router)

# --------------- Import speech router ---------------
from api.routes.speech import speech_router
app.include_router(speech_router)