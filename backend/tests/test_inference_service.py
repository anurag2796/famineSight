import pandas as pd
import pytest

from backend.services.inference import InferenceService
from backend.services.model_registry import registry


class _RF:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, _vector):
        return [[1 - self.p, self.p]]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pcode": "SO2501",
                "date": "2024-01-01",
                "rainfall_anomaly_pct": -8.0,
                "conflict_fatalities": 1,
                "food_price_index": 110.0,
                "ipc_phase4_pct": 8.0,
                "ipc_phase5_pct": 1.0,
                "feat_a": 1.0,
                "feat_b": 2.0,
            },
            {
                "pcode": "SO2501",
                "date": "2024-03-01",
                "rainfall_anomaly_pct": -21.0,
                "conflict_fatalities": 3,
                "food_price_index": 140.0,
                "ipc_phase4_pct": 18.0,
                "ipc_phase5_pct": 4.0,
                "feat_a": 3.0,
                "feat_b": 4.0,
            },
        ]
    )


def test_get_latest_district_data_returns_latest_row(sample_df):
    registry.historical_data = sample_df

    row = InferenceService.get_latest_district_data("SO2501")

    assert row["date"] == "2024-03-01"
    assert row["food_price_index"] == 140.0


def test_get_latest_district_data_raises_when_not_loaded():
    registry.historical_data = pd.DataFrame()

    with pytest.raises(ValueError, match="Historical data not loaded"):
        InferenceService.get_latest_district_data("SO2501")


def test_get_latest_district_data_raises_for_missing_district(sample_df):
    registry.historical_data = sample_df

    with pytest.raises(ValueError, match="No historical data found"):
        InferenceService.get_latest_district_data("SO9999")


@pytest.mark.parametrize(
    "probability, expected_level",
    [
        (0.20, "Low"),
        (0.50, "Medium"),
        (0.90, "High"),
    ],
)
def test_predict_mortality_risk_bands(sample_df, probability, expected_level):
    registry.historical_data = sample_df
    registry.random_forest = _RF(probability)
    registry.classification_metadata = {"feature_columns": ["feat_a", "feat_b"]}

    result = InferenceService.predict_mortality("SO2501")

    assert result.risk_level == expected_level
    assert result.probability == pytest.approx(probability)
    assert result.shap_factors["rainfall_anomaly"] == -21.0


def test_predict_mortality_applies_overrides(sample_df):
    registry.historical_data = sample_df
    registry.random_forest = _RF(0.55)
    registry.classification_metadata = {"feature_columns": ["feat_a", "feat_b"]}

    result = InferenceService.predict_mortality(
        "SO2501", overrides={"rainfall_anomaly_pct": -99.0, "conflict_fatalities": 99}
    )

    assert result.shap_factors["rainfall_anomaly"] == -99.0
    assert result.shap_factors["conflict_fatalities"] == 99


def test_predict_mortality_raises_for_missing_metadata(sample_df):
    registry.historical_data = sample_df
    registry.random_forest = _RF(0.4)
    registry.classification_metadata = {}

    with pytest.raises(ValueError, match="Classification metadata not loaded or empty"):
        InferenceService.predict_mortality("SO2501")


def test_predict_mortality_raises_for_missing_feature_columns(sample_df):
    registry.historical_data = sample_df
    registry.random_forest = _RF(0.4)
    registry.classification_metadata = {"other_key": []}

    with pytest.raises(ValueError, match="Metadata missing 'feature_columns' key"):
        InferenceService.predict_mortality("SO2501")
