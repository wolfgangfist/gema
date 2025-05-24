from run_api import app
import torch
#health_router = APIRouter()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available()
    }