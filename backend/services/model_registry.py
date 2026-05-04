# backend/services/model_registry.py
import joblib
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import HTTPException
import pandas as pd
from src.config import MODELS_DIR, DATA_PROC

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Singleton class for managing machine learning models."""

    _instance = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_all(self):
        """Load all models and data from the models directory."""
        if self._loaded:
            return

        logger.info("Loading models and data...")

        try:
            # Load models
            self.random_forest = joblib.load(MODELS_DIR / "random_forest.joblib")
            self.xgboost_model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
            self.scaler = joblib.load(MODELS_DIR / "scaler.joblib")
            self.pca = joblib.load(MODELS_DIR / "pca.joblib")
            self.kmeans = joblib.load(MODELS_DIR / "kmeans.joblib")
            self.dbscan = joblib.load(MODELS_DIR / "dbscan.joblib")
            self.isolation_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")
            self.lof_model = joblib.load(MODELS_DIR / "lof_model.joblib")

            # Load metadata
            self.classification_metadata = joblib.load(MODELS_DIR / "classification_metadata.joblib")

            # Load analysis results
            self.association_results = self._load_json_file("association_results.json")
            self.cluster_results = self._load_json_file("cluster_results.json")
            self.anomaly_results = self._load_json_file("anomaly_results.json")

            # Load historical data for inference context
            history_path = DATA_PROC / "panel_pca.parquet"
            if history_path.exists():
                self.historical_data = pd.read_parquet(history_path)
            else:
                logger.warning(f"Historical data not found at {history_path}")
                self.historical_data = pd.DataFrame()

            self._loaded = True
            logger.info("All models and data loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise HTTPException(status_code=500, detail="Failed to load models")

    def _load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load JSON file from models directory."""
        try:
            file_path = MODELS_DIR / filename
            if file_path.exists():
                import json
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Could not load {filename}: {e}")
            return {}

    def get_model(self, model_name: str):
        """Get a specific model by name."""
        if not self._loaded:
            self.load_all()

        if hasattr(self, model_name):
            return getattr(self, model_name)
        return None

# Global registry instance
registry = ModelRegistry()