# FamineSight Documentation

## Overview

FamineSight is a comprehensive humanitarian data mining system designed to predict hunger-related mortality in Somalia. The system integrates nine data sources and employs advanced analytics to provide early warning capabilities for humanitarian response teams.

## System Architecture

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
├── scripts/                  # Utility scripts
│   ├── audit.py              # Data audit / quality checks
│   ├── fetch_data.py         # Orchestrate full data fetch
│   ├── generate_notebooks.py # Auto-generate EDA notebooks
│   ├── generate_report_pdf.py# LaTeX report builder
│   └── train_pipeline.py     # Train all ML models
├── src/                      # Core source code
│   ├── config.py             # Centralized configuration
│   ├── data/
│   │   ├── acled_fetcher.py      # ACLED OAuth + conflict data
│   │   ├── chirps_fetcher.py     # CHIRPS rainfall data
│   │   ├── fsnau_fetcher.py      # FSNAU mortality data
│   │   ├── ipc_fetcher.py        # IPC food security phases
│   │   ├── ndvi_fetcher.py       # NDVI vegetation index
│   │   ├── preprocessor.py       # Full data merge & feature pipeline
│   │   ├── shapefile_fetcher.py  # OCHA district shapefiles
│   │   ├── unhcr_fetcher.py      # UNHCR displacement data
│   │   └── wfp_fetcher.py        # WFP food price data
│   ├── analysis/
│   │   ├── anomaly.py            # Isolation Forest, LOF, Z-score
│   │   ├── association.py        # FP-Growth & Apriori rule mining
│   │   ├── classification.py     # Random Forest & XGBoost
│   │   ├── clustering.py         # K-Means & HDBSCAN
│   │   ├── extra_models.py       # Additional model experiments
│   │   └── viz_payload.py        # Visualization data builders
│   └── llm/
│       ├── client.py             # Hybrid Ollama/Groq client
│       ├── groq_client.py        # Groq API client
│       ├── guardrails.py         # AI output validation
│       └── prompts.py            # Prompt templates
├── backend/                  # FastAPI backend
│   ├── Dockerfile
│   ├── main.py               # App entry point + CORS + API key auth
│   ├── requirements.txt
│   ├── security.py           # X-API-Key header auth
│   ├── routers/
│   │   ├── analyze.py        # GET /analyze/rules, /analyze/clusters
│   │   ├── anomaly.py        # GET /anomaly/alerts
│   │   ├── narrative.py      # POST /narrative/generate
│   │   └── predict.py        # POST /predict/mortality
│   ├── schemas/              # Pydantic v2 request/response models
│   └── services/
│       ├── inference.py      # Model inference helpers
│       └── model_registry.py # Loads & caches trained model artifacts
├── frontend/
│   ├── Dockerfile
│   ├── app.py                # Streamlit dashboard
│   └── requirements.txt
├── docs/
│   ├── DATA_SOURCING.md
│   ├── README_ACLED_SETUP.md
│   ├── USAGE_INSTRUCTIONS.md
│   └── wiki/
│       ├── ACLED_INTEGRATION.md
│       ├── DEPLOYMENT.md
│       ├── DEVELOPER_GUIDE.md
│       ├── GROQ_INTEGRATION.md
│       └── README.md         # (this file)
└── report/                   # Academic report (LaTeX source + PDF)
```

## Key Features

### 1. Data Integration (9 Sources)
- **ACLED** — Conflict events, fatalities, civilian targeting (OAuth auth)
- **IPC** — Food security phase percentages (AFI Phases 1–5)
- **WFP** — Food price indices (auto-fetched via HDX)
- **CHIRPS** — Monthly rainfall data (auto-fetched via HDX)
- **NDVI** — Vegetation health index (auto-fetched via HDX)
- **UNHCR** — Internally displaced persons and refugee counts
- **FSNAU** — Crude death rate & under-5 death rate (supplementary)
- **OCHA COD-AB** — Somalia district shapefiles and p-codes (92 districts)

### 2. Analytical Capabilities
- **Association rule mining** — FP-Growth and Apriori algorithms
- **Clustering** — K-Means (k=4), HDBSCAN density-based clustering
- **Classification** — Random Forest with SMOTE, XGBoost (CPU-optimized)
- **Anomaly detection** — Isolation Forest, Local Outlier Factor, Z-score
- **LLM narratives** — AI-generated situation reports with guardrails

### 3. Platform Features
- ARM64 / Jetson AGX Orin optimized (`RF_N_JOBS=4`, `XGB_DEVICE=cpu`)
- Docker-based deployment with health checks
- API key authentication on all endpoints
- Synthetic data fallback for any unavailable source

## Getting Started

### Prerequisites
- Python 3.11+
- Docker and Docker Compose v2
- ACLED credentials (free registration at acleddata.com)
- Generated `API_KEY` (`python -c "import secrets; print(secrets.token_hex(32))"`)

### Quick Start

```bash
# 1. Clone
git clone <repository-url>
cd famineSight

# 2. Configure
cp .env.example .env
# Edit .env: set ACLED_EMAIL, ACLED_PASSWORD, API_KEY

# 3. Fetch data
python scripts/fetch_data.py

# 4. Train models
python scripts/train_pipeline.py

# 5. Start services
docker compose -f docker-compose.dev.yml up --build

# 6. Verify
curl -H "X-API-Key: your_api_key" http://localhost:8000/health
```

## API Endpoints

All endpoints require the `X-API-Key` header.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Backend health check |
| `POST` | `/predict/mortality` | Crisis label + mortality risk prediction |
| `GET` | `/analyze/rules` | Association rules |
| `GET` | `/analyze/clusters` | K-Means district cluster profiles |
| `GET` | `/anomaly/alerts` | Anomaly detection alerts |
| `POST` | `/narrative/generate` | AI-generated situation report |

Interactive API docs: http://localhost:8000/docs

## Model Architecture

### Classification Models
- **Random Forest** — Ensemble method with SMOTE for class balancing; temporal train/val/test split
- **XGBoost** — Gradient boosting with `tree_method="hist"` and `device="cpu"` for ARM64

### Clustering
- **K-Means** — 4-cluster analysis for district vulnerability profiling
- **HDBSCAN** — Density-based clustering for identifying conflict epicenters

### Anomaly Detection
- **Isolation Forest** — Tree-based anomaly scoring
- **Local Outlier Factor** — Density-based anomaly scoring
- **Z-Score Analysis** — Statistical threshold-based alerts

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | ✅ | — | Backend API authentication key |
| `ACLED_EMAIL` | ✅* | — | ACLED account email |
| `ACLED_PASSWORD` | ✅* | — | ACLED account password |
| `OLLAMA_HOST` | — | `http://host.docker.internal:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | — | `qwen3:32b` | Ollama model |
| `GROQ_API_KEY` | — | *(empty)* | Groq API key (optional) |
| `GROQ_MODEL` | — | `llama3-8b-8192` | Groq model |
| `RF_N_JOBS` | — | `4` | RF parallelism (cap at 4 for Jetson) |
| `XGB_DEVICE` | — | `cpu` | XGBoost device |
| `ALLOWED_ORIGINS` | — | `http://localhost:8501,...` | CORS origins |

*Required for real ACLED data; system falls back to synthetic data without them.

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `API_KEY not set` | Add `API_KEY=...` to `.env` |
| ACLED `401` | Verify `ACLED_EMAIL` / `ACLED_PASSWORD` |
| Ollama unreachable | Run `ollama list` on host; check `OLLAMA_HOST` |
| OOM on Jetson | Ensure `RF_N_JOBS=4` and `XGB_DEVICE=cpu` |
| Docker build fails | `docker system prune -f && docker compose build --no-cache` |

## Further Reading

- [ACLED Integration](ACLED_INTEGRATION.md) — OAuth flow details and troubleshooting
- [Groq Integration](GROQ_INTEGRATION.md) — Hybrid LLM architecture
- [Developer Guide](DEVELOPER_GUIDE.md) — Module-level docs and code patterns
- [Deployment Guide](DEPLOYMENT.md) — Production & Jetson deployment steps
- [Data Sourcing](../DATA_SOURCING.md) — Per-source download instructions
- [ACLED Setup](../README_ACLED_SETUP.md) — Quick credential setup guide

## License

This project is for humanitarian and academic purposes only. Unauthorized commercial use is strictly prohibited.