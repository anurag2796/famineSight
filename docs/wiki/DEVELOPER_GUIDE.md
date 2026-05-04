# Developer Guide

## Project Structure

The FamineSight project follows a modular architecture designed for maintainability and scalability:

```
famineSight/
├── .env.example              # Example environment variables
├── README.md                 # Project overview
├── docker-compose.dev.yml    # Docker orchestration (development)
├── pytest.ini                # Test configuration
├── data/
│   ├── raw/                  # Downloaded raw data files
│   └── processed/            # Preprocessed pipeline output
├── models/                   # Trained model artifacts (.pkl)
├── notebooks/                # Jupyter notebooks for EDA
├── scripts/
│   ├── audit.py              # Data audit & quality checks
│   ├── fetch_data.py         # Orchestrate full data fetch
│   ├── generate_notebooks.py # Auto-generate EDA notebooks
│   ├── generate_report_pdf.py# LaTeX report builder
│   └── train_pipeline.py     # Train all ML models
├── src/
│   ├── config.py             # Centralized configuration
│   ├── data/
│   │   ├── acled_fetcher.py
│   │   ├── chirps_fetcher.py
│   │   ├── fsnau_fetcher.py
│   │   ├── ipc_fetcher.py
│   │   ├── ndvi_fetcher.py
│   │   ├── preprocessor.py
│   │   ├── shapefile_fetcher.py
│   │   ├── unhcr_fetcher.py
│   │   └── wfp_fetcher.py
│   ├── analysis/
│   │   ├── anomaly.py
│   │   ├── association.py
│   │   ├── classification.py
│   │   ├── clustering.py
│   │   ├── extra_models.py
│   │   └── viz_payload.py
│   └── llm/
│       ├── client.py
│       ├── groq_client.py
│       ├── guardrails.py
│       └── prompts.py
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   ├── security.py
│   ├── requirements.txt
│   ├── routers/
│   │   ├── analyze.py
│   │   ├── anomaly.py
│   │   ├── narrative.py
│   │   └── predict.py
│   ├── schemas/
│   └── services/
│       ├── inference.py
│       └── model_registry.py
├── frontend/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
└── docs/
    └── wiki/
```

## Configuration Management

### `src/config.py`

The configuration file centralizes all system settings and is the single source of truth:

```python
# src/config.py
from pathlib import Path
import os, csv
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# File paths
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROC      = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
MODELS_DIR     = ROOT / "models"

# ACLED API credentials
ACLED_EMAIL    = os.getenv("ACLED_EMAIL", "")
ACLED_PASSWORD = os.getenv("ACLED_PASSWORD", "")
ACLED_BASE_URL = "https://acleddata.com/api/acled/read"

# Thresholds
CDR_EMERGENCY_THRESHOLD  = 1.0
DROUGHT_ANOMALY_THRESHOLD = -30.0
HIGH_CONFLICT_THRESHOLD  = 10
PRICE_SPIKE_THRESHOLD    = 150

# Historical fetch window
DATA_START_DATE = "2010-01-01"
DATA_END_DATE   = "2024-12-31"

# Jetson-specific configurations
RF_N_JOBS = int(os.getenv("RF_N_JOBS", "4"))
XGB_DEVICE = os.getenv("XGB_DEVICE", "cpu")
RANDOM_STATE = 42
LAG_MONTHS = [1, 2, 3]

# Feature groups
CLIMATE_FEATURES      = ["rainfall_anomaly_pct", "ndvi_anomaly"]
CONFLICT_FEATURES     = ["conflict_events", "conflict_fatalities", "civilian_targeting_events"]
MARKET_FEATURES       = ["food_price_index"]
IPC_FEATURES          = ["ipc_phase1_pct", "ipc_phase2_pct", "ipc_phase3_pct",
                         "ipc_phase4_pct", "ipc_phase5_pct"]
DISPLACEMENT_FEATURES = ["idp_count", "refugee_count"]

ALL_FEATURES = (
    CLIMATE_FEATURES + CONFLICT_FEATURES +
    MARKET_FEATURES + IPC_FEATURES + DISPLACEMENT_FEATURES
)

# Target and auxiliary variables
TARGET_COL   = "crisis_label"
AUX_TARGETS  = ["cdr_per_10k_per_day", "u5dr_per_10k_per_day"]

# IPC crisis threshold: fraction of population in Phase 4+ triggering crisis_label=1
IPC_CRISIS_THRESHOLD = 0.10

# District taxonomy — loaded dynamically from OCHA COD-AB district_lookup.csv
# Falls back to a hardcoded set of 18 regions before first fetch_data.py run
DISTRICT_PCODES: dict  # {district_name: ocha_pcode}
SOMALIA_DISTRICTS: list  # list of district names

# Association rule mining
FP_MIN_SUPPORT        = float(os.getenv("FP_MIN_SUPPORT", "0.005"))
APRIORI_MIN_CONFIDENCE = float(os.getenv("APRIORI_MIN_CONFIDENCE", "0.5"))
APRIORI_MIN_LIFT      = float(os.getenv("APRIORI_MIN_LIFT", "1.0"))

# Clustering
KMEANS_BEST_K = 4

# Classification
RF_N_ESTIMATORS      = 100
XGB_SCALE_POS_WEIGHT = 1.0
SMOTE_K_NEIGHBORS    = 5

# Anomaly detection
ISOFOREST_CONTAMINATION = 0.05
LOF_N_NEIGHBORS         = 20
ZSCORE_THRESHOLD        = 3.0

# LLM
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:32b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# API security — raises ValueError at startup if unset
API_KEY = os.getenv("API_KEY")  # required
```

> **Note:** `API_KEY` is required. The backend raises a `ValueError` at startup if it is not set.

## Data Pipeline

### `src/data/`

| Module | Description |
|--------|-------------|
| `acled_fetcher.py` | OAuth 2.0 token retrieval, paginated fetch, exponential-backoff retry, district-month aggregation |
| `chirps_fetcher.py` | Downloads CHIRPS rainfall CSV via HDX |
| `fsnau_fetcher.py` | Parses FSNAU mortality reports or generates synthetic CDR/U5DR estimates |
| `ipc_fetcher.py` | Fetches IPC food security phase percentages via IPC API |
| `ndvi_fetcher.py` | Downloads NDVI vegetation index data via HDX |
| `preprocessor.py` | Full merge pipeline: load → merge → impute → clip → lag features → scale → PCA → export |
| `shapefile_fetcher.py` | Downloads OCHA COD-AB Somalia admin2 shapefiles and builds `district_lookup.csv` |
| `unhcr_fetcher.py` | Fetches UNHCR IDP and refugee displacement counts |
| `wfp_fetcher.py` | Downloads WFP food price indices from HDX |

### `scripts/fetch_data.py`

Orchestrates the complete data-fetching process:
```bash
python scripts/fetch_data.py           # Real data where available, synthetic fallback
python scripts/fetch_data.py --synthetic  # Force all synthetic data
```

### `scripts/train_pipeline.py`

Runs the full training pipeline:
1. Loads preprocessed `data/processed/merged_data.csv`
2. Trains clustering, classification, and anomaly models
3. Saves artifacts to `models/`

## Analysis Modules

### `src/analysis/`

#### `association.py`
Association rule mining using FP-Growth and Apriori:
- Discretizes continuous features into binary transactions
- Mines frequent itemsets and generates rules
- Filters by confidence and lift thresholds

#### `clustering.py`
District vulnerability profiling:
- K-Means with elbow method (optimal k=4)
- HDBSCAN density-based clustering for conflict epicenters
- District cluster profile computation and labeling

#### `classification.py`
Supervised crisis prediction:
- Temporal train (70%) / validation (15%) / test (15%) split
- Random Forest with SMOTE class balancing
- XGBoost with `tree_method="hist"`, `device="cpu"` (ARM64 safe)
- SHAP feature importance
- Metrics: F1, precision, recall, ROC-AUC

#### `anomaly.py`
Multi-method anomaly detection:
- Isolation Forest (5% contamination rate)
- Local Outlier Factor (20 neighbors)
- Z-score threshold (±3σ)
- Alert generation and severity prioritization

#### `extra_models.py`
Additional model experiments (Gradient Boosting, SVM, etc.)

#### `viz_payload.py`
Builds serializable visualization payloads for the Streamlit frontend (maps, charts, tables).

## LLM Integration

### `src/llm/`

#### `client.py` — HybridClient
The primary LLM interface. Automatically selects the best available backend:
1. **Groq** (cloud) — used if `GROQ_API_KEY` is set and Groq is reachable
2. **Ollama** (local) — used as primary for production / offline mode

Exposes an async generator interface:
```python
from src.llm.client import hybrid_client

async for chunk in hybrid_client.stream("Summarize the famine situation in Bay region."):
    print(chunk, end="", flush=True)
```

#### `groq_client.py`
Implements the Groq OpenAI-compatible streaming API. Reads `GROQ_API_KEY` and `GROQ_MODEL` from config.

#### `prompts.py`
Prompt templates and builder functions for situation reports, alert narratives, and summary generation.

#### `guardrails.py`
Validates LLM output before delivery:
- Blocks responses that mix low-risk language with famine terminology
- Flags probability estimate mismatches
- Enforces a verification disclaimer on all outputs

## Backend API

### `backend/main.py`

FastAPI application with:
- **Lifespan context** — loads ML models on startup via `model_registry`
- **CORS middleware** — origins controlled by `ALLOWED_ORIGINS` env var
- **API key auth** — all routers protected by `Depends(get_api_key)` from `security.py`

### `backend/security.py`

```python
# X-API-Key header required on all protected endpoints
from fastapi.security.api_key import APIKeyHeader
```

### `backend/routers/`

| Router | Endpoints |
|--------|-----------|
| `predict.py` | `POST /predict/mortality` |
| `analyze.py` | `GET /analyze/rules`, `GET /analyze/clusters` |
| `anomaly.py` | `GET /anomaly/alerts` |
| `narrative.py` | `POST /narrative/generate` |

### `backend/services/`

| Service | Description |
|---------|-------------|
| `model_registry.py` | Loads and caches `.pkl` model artifacts from `models/` on startup |
| `inference.py` | Wraps model registry calls for prediction and cluster lookups |

### `backend/schemas/`

Pydantic v2 models for request validation and response serialization. Key schemas:
- `MortalityPredictRequest` / `MortalityPrediction`
- `NarrativeRequest`
- `HealthResponse`, `ClusterProfile`

## Frontend Dashboard

### `frontend/app.py`

Streamlit dashboard with:
- **Crisis Predictor** — District + feature inputs → mortality risk score
- **Vulnerability Map** — Folium choropleth map of district risk levels
- **Association Rules** — Interactive rules explorer
- **Anomaly Alerts** — Real-time alert feed
- **AI Narrative** — LLM-generated situation reports
- **Dark humanitarian theme**

Communicates with the backend via `BACKEND_URL` (default `http://backend:8000`).

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=backend --cov-report=term-missing

# Run a specific module
pytest backend/tests/
```

### Test Strategy
- **Unit tests** — Per-module function tests in `backend/tests/` and `frontend/tests/`
- **Integration tests** — End-to-end API tests against a running backend
- **Synthetic data** — All tests use synthetic data; no real API credentials required

## Development Workflow

### Setting Up

```bash
# 1. Clone and create virtual environment
git clone <repo>
cd famineSight
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: set API_KEY (required), ACLED_EMAIL, ACLED_PASSWORD

# 4. Fetch data and train
python scripts/fetch_data.py --synthetic
python scripts/train_pipeline.py

# 5. Run locally
uvicorn backend.main:app --reload &
streamlit run frontend/app.py
```

### Code Quality Standards

- Python 3.11+; type hints on all public functions
- Comprehensive docstrings (Google style)
- PEP 8 compliance
- Unit tests for all new modules
- Logging via `logging.getLogger(__name__)` — no `print()` in production code

## Jetson AGX Orin Optimization

| Constraint | Setting | Reason |
|-----------|---------|--------|
| Random Forest | `RF_N_JOBS=4` | Prevents OOM on shared GPU/CPU RAM |
| XGBoost | `XGB_DEVICE=cpu` | ARM64 CUDA support is limited |
| Backend memory | `4G` Docker limit | Leaves headroom for Ollama |
| Frontend memory | `1G` Docker limit | Streamlit is lightweight |
| GDAL | Installed in Dockerfile | Required for shapefile parsing |

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `API_KEY not set` | Add `API_KEY=...` to `.env` |
| `401` on API calls | Pass `-H "X-API-Key: your_key"` in requests |
| ACLED `401` | Verify `ACLED_EMAIL` / `ACLED_PASSWORD` |
| Model not found | Run `scripts/train_pipeline.py` first |
| OOM on Jetson | Ensure `RF_N_JOBS=4`, `XGB_DEVICE=cpu` in `.env` |
| Docker build fails | `docker system prune -f`, rebuild with `--no-cache` |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for new functionality
4. Follow coding standards (type hints, docstrings, PEP 8)
5. Submit a pull request with a clear description

## Support

For support, please open an issue in the repository.