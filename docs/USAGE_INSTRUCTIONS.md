# FamineSight Usage Instructions

## System Overview

FamineSight is a humanitarian data mining system designed to predict hunger-related mortality in Somalia. It integrates nine data sources and employs unsupervised analytics, supervised machine learning, and LLM-based narrative generation to provide early warning capabilities for humanitarian response teams.

## System Architecture

```
famineSight/
├── src/                      # Core source code
│   ├── config.py             # Centralized configuration
│   ├── data/                 # Data fetching & preprocessing
│   │   ├── acled_fetcher.py
│   │   ├── chirps_fetcher.py
│   │   ├── fsnau_fetcher.py
│   │   ├── ipc_fetcher.py
│   │   ├── ndvi_fetcher.py
│   │   ├── preprocessor.py
│   │   ├── shapefile_fetcher.py
│   │   ├── unhcr_fetcher.py
│   │   └── wfp_fetcher.py
│   ├── analysis/             # ML & analytics modules
│   │   ├── anomaly.py
│   │   ├── association.py
│   │   ├── classification.py
│   │   ├── clustering.py
│   │   ├── extra_models.py
│   │   └── viz_payload.py
│   └── llm/                  # LLM integration
│       ├── client.py         # Hybrid Ollama/Groq client
│       ├── groq_client.py
│       ├── guardrails.py
│       └── prompts.py
├── backend/                  # FastAPI backend
│   ├── main.py               # App entry point
│   ├── security.py           # API key auth
│   ├── routers/
│   ├── schemas/
│   └── services/
├── frontend/
│   └── app.py                # Streamlit dashboard
└── data/
    ├── raw/                  # Downloaded raw data files
    └── processed/            # Pipeline output
```

## Configuration

### Environment Variables (`.env` file)

```bash
# ─── ACLED Credentials ─────────────────────────────────────────────────────
ACLED_EMAIL=your_email@organization.org
ACLED_PASSWORD=your_acled_password

# ─── LLM Configuration ─────────────────────────────────────────────────────
# Ollama runs on the host; accessible inside Docker via host.docker.internal
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:32b

# Groq (optional cloud fallback — leave empty to use Ollama only)
GROQ_API_KEY=
GROQ_MODEL=llama3-8b-8192

# ─── API Security ───────────────────────────────────────────────────────────
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
API_KEY=replace_with_a_strong_random_key

# ─── Service URLs ───────────────────────────────────────────────────────────
BACKEND_URL=http://backend:8000
ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501,http://frontend:8501

# ─── Hardware Tuning ────────────────────────────────────────────────────────
RF_N_JOBS=4
XGB_DEVICE=cpu
```

> **Required:** `API_KEY` must be set. All API endpoints require the `X-API-Key` header.

## Running the System

### Option 1: Docker Compose (Recommended)

```bash
# Development mode (includes Ollama sidecar, ~4 GB backend memory)
docker compose -f docker-compose.dev.yml up --build

# Verify the backend is healthy
curl -H "X-API-Key: your_api_key" http://localhost:8000/health
```

### Option 2: Local Development (without Docker)

```bash
# Install dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Start the backend
uvicorn backend.main:app --reload

# In a separate terminal, start the frontend
streamlit run frontend/app.py
```

### LLM Configuration

| Mode | Config | Description |
|------|--------|-------------|
| **Ollama only** (default) | `GROQ_API_KEY=` (empty) | Fully offline, requires Ollama running on host |
| **Groq only** | `GROQ_API_KEY=your_key` | Cloud inference, faster for development |
| **Hybrid** (recommended) | Both set | Uses Groq when available, falls back to Ollama |

## Data Pipeline

### Data Sources

| Source | Data | Notes |
|--------|------|-------|
| ACLED | Conflict events, fatalities, civilian targeting | OAuth auth required |
| WFP | Food price indices | Auto-fetched via HDX (no key needed) |
| CHIRPS | Monthly rainfall (mm) | Auto-fetched via HDX |
| NDVI | Vegetation health index | Auto-fetched via HDX |
| UNHCR | IDP count, refugee count | Auto-fetched via API |
| IPC | Food security phase percentages | Auto-fetched via IPC API |
| FSNAU | Crude death rate, under-5 death rate | Supplementary / sparse |
| OCHA COD-AB | Somalia district shapefiles & p-codes | Auto-fetched |

### Running the Pipeline

```bash
# Fetch all data (real + synthetic fallback)
python scripts/fetch_data.py

# Force synthetic data (no credentials required)
python scripts/fetch_data.py --synthetic

# Train ML models on processed data
python scripts/train_pipeline.py
```

### Processing Steps

1. **Data Fetching** — Real API data or synthetic fallback
2. **Merging** — Join all sources on district × month
3. **Imputation** — Forward-fill then median fill missing values
4. **Outlier Clipping** — 1st–99th percentile clipping
5. **Lag Feature Engineering** — 1-, 2-, 3-month lags for key indicators
6. **Feature Scaling** — StandardScaler on training split
7. **PCA** — Optional dimensionality reduction
8. **Temporal Sorting** — Chronological train/validation/test split

## API Endpoints

All endpoints require the `X-API-Key: <your_api_key>` header.

### Prediction
- `POST /predict/mortality` — Predict crisis label and mortality risk for a district

### Analysis
- `GET /analyze/rules` — Get association rules (FP-Growth / Apriori)
- `GET /analyze/clusters` — Get K-Means district cluster profiles

### Anomaly Detection
- `GET /anomaly/alerts` — Get anomaly alerts (Isolation Forest, LOF, Z-score)

### Narrative Generation
- `POST /narrative/generate` — Generate AI situation report (Ollama / Groq)

### System
- `GET /health` — Backend health check (model load status, Ollama availability)

## LLM Integration

### Hybrid Client Architecture

`src/llm/client.py` implements a `HybridClient` that:
1. Prefers **local Ollama** (secure, offline, primary for production)
2. Falls back to **Groq API** (cloud, faster, useful for development)
3. Applies **guardrails** (`src/llm/guardrails.py`) to validate AI output before delivery

### Python Usage

```python
from src.llm.client import hybrid_client

# Async streaming (works with both Ollama and Groq)
async for chunk in hybrid_client.stream("Summarize the crisis situation in Baidoa."):
    print(chunk, end="", flush=True)
```

## Testing

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=src --cov=backend

# Docker integration test
docker compose -f docker-compose.dev.yml up -d
curl -H "X-API-Key: your_api_key" http://localhost:8000/health
```

## Production Deployment

### Prerequisites
- Python 3.11+, Docker & Docker Compose v2
- ACLED credentials and a generated `API_KEY`
- For Jetson AGX Orin: 60+ GB RAM, JetPack 6.x

### Steps

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env: set ACLED_EMAIL, ACLED_PASSWORD, API_KEY, etc.

# 2. Fetch data and train models
python scripts/fetch_data.py
python scripts/train_pipeline.py

# 3. Build and start services
docker compose -f docker-compose.dev.yml build
docker compose -f docker-compose.dev.yml up -d

# 4. Verify
curl -H "X-API-Key: $API_KEY" http://localhost:8000/health
```

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `API_KEY environment variable is not set` | Generate and add `API_KEY=...` to `.env` |
| ACLED `401 Unauthorized` | Check `ACLED_EMAIL` and `ACLED_PASSWORD` in `.env` |
| LLM not responding | Verify Ollama is running: `ollama list` |
| Memory OOM on Jetson | Ensure `RF_N_JOBS=4` and `XGB_DEVICE=cpu` in `.env` |
| Docker build failure | Run `docker system prune -f` then rebuild with `--no-cache` |

### Fallback Behavior
- If ACLED API fails → synthetic conflict data is used
- If Groq API fails → Ollama is used for narrative generation
- If Ollama is unavailable → narrative endpoint returns an error

## Security

- All credentials stored in `.env` (set permissions: `chmod 600 .env`)
- All API endpoints protected by `X-API-Key` header authentication
- No credentials hardcoded anywhere in source code
- Groq is optional — production deployments should use Ollama for full offline operation

## License

This project is for humanitarian and academic purposes only. Unauthorized commercial use is strictly prohibited.