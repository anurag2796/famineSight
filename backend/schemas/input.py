# backend/schemas/input.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MortalityPredictRequest(BaseModel):
    """Request model for mortality prediction."""
    district_pcode: str = Field(..., description="District pcode (e.g., SO0001)")
    rainfall_anomaly_pct: Optional[float] = Field(None, description="Rainfall anomaly percentage")
    conflict_fatalities: Optional[int] = Field(None, description="Number of conflict fatalities")
    food_price_index: Optional[float] = Field(None, description="Food price index")
    ipc_phase4_pct: Optional[float] = Field(None, description="IPC phase 4 percentage")
    ipc_phase5_pct: Optional[float] = Field(None, description="IPC phase 5 percentage")

    @field_validator('district_pcode')
    @classmethod
    def validate_pcode(cls, v):
        if not v or len(v) < 6:
            raise ValueError('District pcode must be at least 6 characters')
        return v

    @field_validator('conflict_fatalities')
    @classmethod
    def validate_conflict_fatalities(cls, v):
        if v is not None and v < 0:
            raise ValueError('Conflict fatalities cannot be negative')
        return v

    @field_validator('food_price_index')
    @classmethod
    def validate_food_price_index(cls, v):
        if v is not None and v < 0:
            raise ValueError('Food price index cannot be negative')
        return v

class NarrativeRequest(BaseModel):
    """Request model for narrative generation."""
    prediction: MortalityPredictRequest
    alerts: list = Field(default=[], description="List of alerts to include in narrative")
    rules: dict = Field(default={}, description="Association rules to include in narrative")