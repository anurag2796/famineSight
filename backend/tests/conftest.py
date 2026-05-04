import os
from typing import Any, Dict

import pandas as pd
import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("API_KEY", "test-api-key")


class DummyRFModel:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def predict_proba(self, _vector):
        return [[1.0 - self.probability, self.probability]]


@pytest.fixture
def sample_historical_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pcode": "SO2501",
                "date": "2024-01-01",
                "rainfall_anomaly_pct": -5.0,
                "conflict_fatalities": 2,
                "food_price_index": 120.0,
                "ipc_phase4_pct": 10.0,
                "ipc_phase5_pct": 2.0,
                "feat_a": 1.2,
                "feat_b": 3.4,
            },
            {
                "pcode": "SO2501",
                "date": "2024-02-01",
                "rainfall_anomaly_pct": -15.0,
                "conflict_fatalities": 5,
                "food_price_index": 145.0,
                "ipc_phase4_pct": 14.0,
                "ipc_phase5_pct": 3.0,
                "feat_a": 2.2,
                "feat_b": 4.4,
            },
            {
                "pcode": "SO2601",
                "date": "2024-02-01",
                "rainfall_anomaly_pct": 4.0,
                "conflict_fatalities": 0,
                "food_price_index": 100.0,
                "ipc_phase4_pct": 3.0,
                "ipc_phase5_pct": 0.0,
                "feat_a": 1.0,
                "feat_b": 1.0,
            },
        ]
    )


@pytest.fixture
def api_client(monkeypatch, sample_historical_data: pd.DataFrame):
    from backend import main

    monkeypatch.setattr(main.registry, "load_all", lambda: None)
    main.registry._loaded = True
    main.registry.random_forest = DummyRFModel(probability=0.82)
    main.registry.classification_metadata = {"feature_columns": ["feat_a", "feat_b"]}
    main.registry.historical_data = sample_historical_data.copy()
    main.registry.association_results = {
        "fpgrowth": [{"rule": "A->B"}],
        "apriori": [{"rule": "C->D"}],
    }
    main.registry.cluster_results = {
        "district_profiles": [
            {
                "pcode": "SO2501",
                "district": "Baidoa",
                "kmeans_cluster": 2,
                "cluster_name": "High Concern",
                "rainfall_anomaly_pct": -15.0,
                "conflict_fatalities": 5,
            }
        ]
    }
    main.registry.anomaly_results = {
        "alerts": [{"district": "Baidoa", "severity": "CRITICAL"}]
    }

    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def auth_headers() -> Dict[str, Any]:
    return {"X-API-Key": os.environ["API_KEY"]}
