from fastapi import APIRouter
import torch
health_router = APIRouter()

@health_router.get("/health")
def health_check():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available()
    }