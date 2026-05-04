# src/config.py
from pathlib import Path
import os
import csv
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# File paths
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
MODELS_DIR = ROOT / "models"

# ACLED API credentials
ACLED_EMAIL = os.getenv("ACLED_EMAIL", "")
ACLED_PASSWORD = os.getenv("ACLED_PASSWORD", "")
ACLED_BASE_URL = "https://acleddata.com/api/acled/read"

# Thresholds
CDR_EMERGENCY_THRESHOLD = 1.0
DROUGHT_ANOMALY_THRESHOLD = -30.0
HIGH_CONFLICT_THRESHOLD = 10
PRICE_SPIKE_THRESHOLD = 150

# Jetson-specific configurations
RF_N_JOBS = int(os.getenv("RF_N_JOBS", "4"))
if RF_N_JOBS == -1:
    import warnings
    warnings.warn("RF_N_JOBS=-1 will cause OOM on Jetson AGX Orin. Setting to 4.", RuntimeWarning)
    RF_N_JOBS = 4

XGB_DEVICE = os.getenv("XGB_DEVICE", "cpu")
RANDOM_STATE = 42
LAG_MONTHS = [1, 2, 3]

# Historical fetch window — extended back to include 2011 famine
DATA_START_DATE = "2010-01-01"
DATA_END_DATE   = "2024-12-31"

# Feature column lists
CLIMATE_FEATURES = [
    "rainfall_anomaly_pct",
    "ndvi_anomaly",         # vegetation health (MODIS/NDVI)
]

CONFLICT_FEATURES = [
    "conflict_events",
    "conflict_fatalities",
    "civilian_targeting_events"
]

MARKET_FEATURES = [
    "food_price_index"
]

IPC_FEATURES = [
    "ipc_phase1_pct",
    "ipc_phase2_pct",
    "ipc_phase3_pct",
    "ipc_phase4_pct",
    "ipc_phase5_pct"
]

DISPLACEMENT_FEATURES = [
    "idp_count",            # internally displaced persons (UNHCR)
    "refugee_count",        # refugees/asylum seekers (UNHCR)
]

ALL_FEATURES = (
    CLIMATE_FEATURES
    + CONFLICT_FEATURES
    + MARKET_FEATURES
    + IPC_FEATURES
    + DISPLACEMENT_FEATURES
)

# Target and auxiliary variables
TARGET_COL = "crisis_label"
# FSNAU mortality columns — sparse/supplementary, kept as aux targets not main features
AUX_TARGETS = ["cdr_per_10k_per_day", "u5dr_per_10k_per_day"]

# IPC crisis threshold: fraction of population in Phase 4+ that triggers crisis_label=1
IPC_CRISIS_THRESHOLD = 0.10  # 10% of population in Emergency or Catastrophe

# ---------------------------------------------------------------------------
# District taxonomy — loaded dynamically from the OCHA COD-AB district_lookup.
# Uses real OCHA admin2 pcodes (e.g. SO2201) across all 92 Somali districts.
# Falls back to a minimal hardcoded set if the lookup CSV is not yet present
# (i.e. before the first `fetch_data.py` run).
# ---------------------------------------------------------------------------

_LOOKUP_CSV = ROOT / "data" / "raw" / "shapefiles" / "district_lookup.csv"

def _load_district_pcodes() -> dict:
    """Return {district_name: ocha_pcode} for all districts in the lookup CSV."""
    if not _LOOKUP_CSV.exists():
        # Minimal fallback covering the 20 key zones used before shapefile fetch
        return {
            "Banadir":      "SO22",
            "Bay":          "SO24",
            "Gedo":         "SO26",
            "Hiraan":       "SO20",
            "Lower Juba":   "SO28",
            "Lower Shabelle": "SO23",
            "Middle Juba":  "SO27",
            "Middle Shabelle": "SO21",
            "Mudug":        "SO18",
            "Galgaduud":    "SO19",
            "Sanaag":       "SO15",
            "Togdheer":     "SO13",
            "Sool":         "SO14",
            "Woqooyi Galbeed": "SO12",
            "Bari":         "SO16",
            "Nugaal":       "SO17",
            "Bakool":       "SO25",
            "Awdal":        "SO11",
        }
    result = {}
    with open(_LOOKUP_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name  = row.get("name", "").strip()
            pcode = row.get("pcode", "").strip()
            if name and pcode and pcode != "Unspecified":
                result[name] = pcode
    return result

DISTRICT_PCODES: dict = _load_district_pcodes()
SOMALIA_DISTRICTS: list = list(DISTRICT_PCODES.keys())

# Association rule mining parameters
# Can be overridden via environment variable `FP_MIN_SUPPORT` for tuning
FP_MIN_SUPPORT = float(os.getenv("FP_MIN_SUPPORT", "0.005"))
APRIORI_MIN_CONFIDENCE = float(os.getenv("APRIORI_MIN_CONFIDENCE", "0.5"))
APRIORI_MIN_LIFT = float(os.getenv("APRIORI_MIN_LIFT", "1.0"))

# Clustering parameters
KMEANS_BEST_K = 4

# Classification parameters
RF_N_ESTIMATORS = 100
XGB_SCALE_POS_WEIGHT = 1.0
SMOTE_K_NEIGHBORS = 5

# Anomaly detection parameters
ISOFOREST_CONTAMINATION = 0.05
LOF_N_NEIGHBORS = 20
ZSCORE_THRESHOLD = 3.0

# LLM parameters
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:32b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
_api_key = os.getenv("API_KEY", "")
if not _api_key:
    raise ValueError(
        "API_KEY environment variable is not set. "
        "Generate a strong random key and add it to your .env file."
    )
API_KEY = _api_key