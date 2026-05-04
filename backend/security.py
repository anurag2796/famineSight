# backend/security.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.config import API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not API_KEY:
        # If no API key is configured, allow all (for development)
        return None
        
    if api_key_header == API_KEY:
        return api_key_header
        
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )
