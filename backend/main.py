# backend/main.py
import uvicorn
import logging
import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Ensure project root is on the path so both 'src' and 'backend' are importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from backend.schemas.input import MortalityPredictRequest, NarrativeRequest
from backend.schemas.output import MortalityPrediction, HealthResponse, ClusterProfile
from backend.services.model_registry import registry
from backend.routers import predict, analyze, anomaly, narrative

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    logger.info("Starting FamineSight backend...")

    # Load models on startup
    try:
        registry.load_all()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    logger.info("Shutting down FamineSight backend...")

# Create FastAPI app
app = FastAPI(
    title="FamineSight API",
    description="API for humanitarian data mining and famine prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Build CORS allowed origins from env var or safe defaults
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,http://frontend:8501"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.security import get_api_key  # noqa: E402 (after sys.path setup)

# Include routers
app.include_router(predict.router, prefix="/predict", tags=["prediction"], dependencies=[Depends(get_api_key)])
app.include_router(analyze.router, prefix="/analyze", tags=["analysis"], dependencies=[Depends(get_api_key)])
app.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"], dependencies=[Depends(get_api_key)])
app.include_router(narrative.router, prefix="/narrative", tags=["narrative"], dependencies=[Depends(get_api_key)])

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if models are loaded
        models_loaded = registry._loaded

        from datetime import datetime, timezone
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://host.docker.internal:11434/api/tags", timeout=2.0)
                ollama_available = response.status_code == 200
        except Exception:
            ollama_available = False

        return HealthResponse(
            status="healthy",
            model_available=models_loaded,
            data_available=True,
            ollama_available=ollama_available,
            timestamp=datetime.now(timezone.utc)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )