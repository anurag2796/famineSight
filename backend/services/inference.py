import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from backend.services.model_registry import registry
from backend.schemas.output import MortalityPrediction, ClusterProfile
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class InferenceService:
    @staticmethod
    def get_latest_district_data(district_pcode: str) -> pd.Series:
        historical_data = getattr(registry, 'historical_data', None)
        if historical_data is None or historical_data.empty:
            raise ValueError("Historical data not loaded in registry")
            
        district_data = historical_data[historical_data['pcode'] == district_pcode]
        if district_data.empty:
            raise ValueError(f"No historical data found for district {district_pcode}")
            
        return district_data.sort_values('date').iloc[-1].copy()

    @staticmethod
    def predict_mortality(district_pcode: str, overrides: Dict[str, float] = None) -> MortalityPrediction:
        rf_model = registry.random_forest
        metadata = registry.classification_metadata
        
        # Defensive validation: ensure critical metadata keys exist
        if not metadata:
            raise ValueError("Classification metadata not loaded or empty")
        if 'feature_columns' not in metadata:
            raise ValueError(f"Metadata missing 'feature_columns' key. Available keys: {list(metadata.keys())}")
        
        feature_cols = metadata['feature_columns']
        
        latest_row = InferenceService.get_latest_district_data(district_pcode)
        
        if overrides:
            for col, val in overrides.items():
                if val is not None:
                    latest_row[col] = val
                    
        feature_vector = []
        for col in feature_cols:
            if col in latest_row:
                feature_vector.append(latest_row[col])
            else:
                feature_vector.append(0.0)
                
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        prediction = rf_model.predict_proba(feature_vector)[0]
        probability = float(prediction[1])
        
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        shap_factors = {
            "rainfall_anomaly": latest_row.get('rainfall_anomaly_pct', 0),
            "conflict_fatalities": latest_row.get('conflict_fatalities', 0),
            "food_price_index": latest_row.get('food_price_index', 0),
            "ipc_phase4": latest_row.get('ipc_phase4_pct', 0),
            "ipc_phase5": latest_row.get('ipc_phase5_pct', 0)
        }
        
        return MortalityPrediction(
            district_pcode=district_pcode,
            risk_level=risk_level,
            probability=probability,
            confidence=1.0,
            shap_factors=shap_factors,
            timestamp=datetime.now(timezone.utc)
        )

inference_service = InferenceService()
