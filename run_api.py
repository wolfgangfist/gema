import os

from fastapi import FastAPI

# --------------- Init FastAPI app ---------------
app = FastAPI()

INTERNAL_PORT = os.getenv("INTERNAL_PORT", 8888)

@app.on_event("startup")
async def startup_event():
    print(f"✅ CSM API is live and listening on http://0.0.0.0:{INTERNAL_PORT}/")

@app.get("/")
def root():
    return {"status": "✅ CSM API is running"}

import api.routes.health

# --------------- Import health router ---------------
#from api.routes.health import health_router
#app.include_router(health_router)

# --------------- Import speech router ---------------
from api.routes.speech import speech_router
app.include_router(speech_router)