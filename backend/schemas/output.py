# backend/schemas/output.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

class MortalityPrediction(BaseModel):
    """Response model for mortality prediction."""
    district_pcode: str
    risk_level: str  # Low, Medium, High
    probability: float
    confidence: float
    shap_factors: Dict[str, float]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_available: bool = False
    data_available: bool = False
    ollama_available: bool = False

class ClusterProfile(BaseModel):
    """Response model for cluster profiles."""
    pcode: str
    district: str
    kmeans_cluster: int
    cluster_name: str
    features: Dict[str, Any]