# backend/routers/predict.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging
import numpy as np
from src.config import RF_N_JOBS
from backend.schemas.input import MortalityPredictRequest
from backend.schemas.output import MortalityPrediction
from backend.services.model_registry import registry

import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/mortality", response_model=MortalityPrediction)
async def predict_mortality(request: MortalityPredictRequest):
    """
    Predict mortality risk for a district.
    """
    try:
        # Get the loaded models
        rf_model = registry.random_forest
        metadata = registry.classification_metadata
        feature_cols = metadata['feature_columns']
        historical_data = getattr(registry, 'historical_data', None)

        if historical_data is None or historical_data.empty:
            raise ValueError("Historical data not loaded in registry")

        # Get the most recent row for this district
        district_data = historical_data[historical_data['pcode'] == request.district_pcode]
        if district_data.empty:
            raise ValueError(f"No historical data found for district {request.district_pcode}")
            
        latest_row = district_data.sort_values('date').iloc[-1].copy()

        # Apply overrides from the request
        overrides = {
            'rainfall_anomaly_pct': request.rainfall_anomaly_pct,
            'conflict_fatalities': request.conflict_fatalities,
            'food_price_index': request.food_price_index,
            'ipc_phase4_pct': request.ipc_phase4_pct,
            'ipc_phase5_pct': request.ipc_phase5_pct
        }

        for col, val in overrides.items():
            if val is not None:
                latest_row[col] = val

        # Extract exactly the features the model expects
        feature_vector = []
        for col in feature_cols:
            if col in latest_row:
                feature_vector.append(latest_row[col])
            else:
                feature_vector.append(0.0)  # Fallback
                
        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Make prediction directly on the vector (already scaled/transformed context)
        prediction = rf_model.predict_proba(feature_vector)[0]
        probability = float(prediction[1])  # Probability of crisis (class 1)

        # Get risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Get SHAP factors (simplified)
        shap_factors = {
            "rainfall_anomaly": latest_row.get('rainfall_anomaly_pct', 0),
            "conflict_fatalities": latest_row.get('conflict_fatalities', 0),
            "food_price_index": latest_row.get('food_price_index', 0),
            "ipc_phase4": latest_row.get('ipc_phase4_pct', 0),
            "ipc_phase5": latest_row.get('ipc_phase5_pct', 0)
        }

        return MortalityPrediction(
            district_pcode=request.district_pcode,
            risk_level=risk_level,
            probability=probability,
            confidence=1.0,
            shap_factors=shap_factors
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")