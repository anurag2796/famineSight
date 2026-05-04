# FamineSight 🌍

**Predicting hunger-related mortality in Somalia using AI.** FamineSight is an end-to-end famine early warning system that combines traditional econometric models with cutting-edge AI to provide timely alerts and decision support for humanitarian action.

## Features

- **Multi-Source Data Integration** — ACLED conflict, WFP food prices, CHIRPS rainfall, NDVI vegetation, UNHCR displacement, IPC phases, and FSNAU mortality
- **Predictive Analytics** — Forecast food security crises with Random Forest and XGBoost
- **Unsupervised Analytics** — K-Means clustering, HDBSCAN, association rule mining, and anomaly detection
- **Hybrid AI Narratives** — Groq cloud + local Ollama for context-aware situation reporting
- **Full Pipeline** — From data ingestion to prediction API
- **Production-Ready** — Docker support, API key security, and FastAPI backend

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/famineSight.git
cd famineSight
```

### 2. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 3. Set Up Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` with your credentials:

```bash
# ACLED API Credentials (Required for real conflict data)
ACLED_EMAIL=your_email@example.com
ACLED_PASSWORD=your_acled_password

# LLM Configuration
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:32b
GROQ_API_KEY=your_groq_api_key        # Optional cloud fallback
GROQ_MODEL=llama3-8b-8192

# API Security (required — generate a strong random key)
API_KEY=replace_with_a_strong_random_key

# Performance Settings
RF_N_JOBS=4
XGB_DEVICE=cpu
```

Generate a secure `API_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Fetch Data

```bash
# Fetch all data sources (falls back to synthetic if credentials are unavailable)
python scripts/fetch_data.py

# Force synthetic data (no credentials required)
python scripts/fetch_data.py --synthetic
```

This will download and process:
1. IPC food security phase data
2. ACLED conflict events
3. WFP food price indices
4. CHIRPS rainfall data
5. NDVI vegetation index data
6. UNHCR displacement data
7. Somalia district shapefiles
8. FSNAU mortality estimates

Processed data is saved to `data/processed/`.

### 5. Train Models

```bash
python scripts/train_pipeline.py
```

Trained model artifacts are saved to `models/`.

### 6. Run the API

Start the FastAPI backend:

```bash
uvicorn backend.main:app --reload
```

Then access:
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Predict Mortality: `POST /predict/mortality`

> **Note:** All API endpoints require the `X-API-Key` header set to your `API_KEY` value.

### 7. Run the Dashboard

```bash
streamlit run frontend/app.py
```

Access the dashboard at: http://localhost:8501

## Project Structure

```
famineSight/
├── backend/                  # FastAPI backend
│   ├── Dockerfile
│   ├── main.py               # Application entry point
│   ├── requirements.txt
│   ├── routers/              # API route handlers
│   │   ├── analyze.py        # Association rules & clusters
│   │   ├── anomaly.py        # Anomaly alerts
│   │   ├── narrative.py      # AI narrative generation
│   │   └── predict.py        # Mortality prediction
│   ├── schemas/              # Pydantic v2 data models
│   ├── security.py           # API key authentication
│   └── services/             # Business logic & model registry
├── frontend/
│   ├── Dockerfile
│   ├── app.py                # Streamlit dashboard
│   └── requirements.txt
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
├── scripts/                  # Utility scripts
│   ├── audit.py
│   ├── fetch_data.py
│   ├── generate_notebooks.py
│   ├── generate_report_pdf.py
│   └── train_pipeline.py
├── data/                     # Data storage
│   ├── raw/                  # Downloaded raw data
│   └── processed/            # Pipeline output
├── models/                   # Trained model artifacts
├── notebooks/                # Research & EDA notebooks
├── docs/                     # Documentation
│   ├── DATA_SOURCING.md
│   ├── README_ACLED_SETUP.md
│   ├── USAGE_INSTRUCTIONS.md
│   └── wiki/
└── report/                   # Academic report (LaTeX)
```

## Configuration

All configuration is driven by `.env` and loaded via `src/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACLED_EMAIL` | — | ACLED account email |
| `ACLED_PASSWORD` | — | ACLED account password |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Local Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3:32b` | Ollama model name |
| `GROQ_API_KEY` | *(empty)* | Groq API key (optional) |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model name |
| `API_KEY` | *(required)* | Backend API authentication key |
| `RF_N_JOBS` | `4` | Random Forest parallelism (cap at 4 for Jetson) |
| `XGB_DEVICE` | `cpu` | XGBoost device (ARM64 requires `cpu`) |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS origins for the API |

## Model Architecture

```mermaid
graph TD
    subgraph "Data Layer"
        IPC[IPC Data] --> Preprocessor[Data Preprocessor]
        ACLED[ACLED Conflict] --> Preprocessor
        WFP[WFP Food Prices] --> Preprocessor
        CHIRPS[CHIRPS Rainfall] --> Preprocessor
        NDVI[NDVI Vegetation] --> Preprocessor
        UNHCR[UNHCR Displacement] --> Preprocessor
        FSNAU[FSNAU Mortality] --> Preprocessor
    end

    subgraph "Analytics Layer"
        Preprocessor --> Classification[Classification]
        Preprocessor --> Clustering[Clustering]
        Preprocessor --> Association[Association Rules]
        Preprocessor --> Anomaly[Anomaly Detection]

        subgraph "Classification"
            Classification --> RF[Random Forest]
            Classification --> XGB[XGBoost]
        end

        subgraph "Clustering"
            Clustering --> KMeans[K-Means]
            Clustering --> HDBSCAN[HDBSCAN]
        end
    end

    subgraph "LLM Layer"
        Classification --> HybridClient[Hybrid LLM Client]
        HybridClient --> Ollama[Ollama / qwen3:32b]
        HybridClient --> Groq[Groq / llama3-8b-8192]
    end

    subgraph "Application Layer"
        HybridClient --> API[FastAPI Backend]
        API --> Dashboard[Streamlit Dashboard]
    end
```

## Production Deployment

### Docker Deployment

```bash
# Development (includes Ollama sidecar, smaller memory limits)
docker compose -f docker-compose.dev.yml up --build

# Check service health
curl -s http://localhost:8000/health
```

Access the application:
- Frontend dashboard: http://localhost:8501
- API: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs

### Production Configuration

For production deployment, ensure:
- `API_KEY` is a strong random secret (64-character hex)
- Real ACLED credentials are configured
- Groq API key is set (optional but recommended)
- Sufficient RAM (16 GB+ for x86; 60 GB+ for Jetson AGX Orin)
- `ALLOWED_ORIGINS` lists only your trusted frontend origins

## AI Model Settings

| Component | Model | Purpose |
|-----------|-------|---------|
| **Clustering** | K-Means (k=4), HDBSCAN | Unsupervised district vulnerability profiling |
| **Classification** | Random Forest, XGBoost | Supervised crisis prediction |
| **Groq LLM** | llama3-8b-8192 (configurable) | Cloud fallback for narrative generation |
| **Ollama LLM** | qwen3:32b (configurable) | Primary local narrative & alerting |

## Troubleshooting

**1. `API_KEY` not set error:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
# Add the output as API_KEY= in your .env file
```

**2. LLM connection issues:**
```bash
# Check Ollama is running on the host
ollama list
curl http://localhost:11434/api/tags
```

**3. ACLED authentication errors:**
```bash
# Verify credentials in .env
grep ACLED .env
python -c "from src.data.acled_fetcher import get_acled_token; print(get_acled_token())"
```

**4. Memory issues on Jetson:**
```bash
# Ensure these are set in .env
RF_N_JOBS=4
XGB_DEVICE=cpu
```

## License

This project is for humanitarian and academic purposes only. Unauthorized commercial use is strictly prohibited.

## Support

For questions or issues, please open a GitHub issue in the repository.