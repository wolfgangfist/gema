import os

from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader

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