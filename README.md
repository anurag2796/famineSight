# FamineSight 🌍

**Predicting hunger-related mortality in Somalia using AI.** FamineSight is an end-to-end famine early warning system that combines traditional econometric models with cutting-edge AI to provide timely alerts and decision support for humanitarian action.

## Features

- **30-Second Health Check** - Instant pipeline validation
- **Real-Time Monitoring** - Continuously monitor climate and conflict
- **Predictive Analytics** - Forecast food security with ML models
- **Hybrid AI** - Groq + Ollama for context-aware alerting
- **Full Pipeline** - From data ingestion to prediction API
- **Production-Ready** - Docker support and FastAPI backend

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/famineSight.git
cd famineSight
```

### 2. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` with your credentials:

```bash
# ACLED API Credentials (Required for real data)
ACLED_EMAIL=your_email@example.com
ACLED_PASSWORD=your_password

# LLM Configuration
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:32b
GROQ_API_KEY=your_groq_api_key

# Performance Settings
RF_N_JOBS=4
XGB_DEVICE=cpu
```

### 4. Run the Pipeline

**Quick Health Check (30 seconds):**

```bash
python src/quick_health_check.py
```

**Full Data Pipeline:**

```bash
python src/data/preprocessor.py --synthetic
```

This will:
1. Download and process IPC data
2. Fetch ACLED conflict data
3. Merge climate and conflict data
4. Train ML models
5. Generate synthetic test data
6. Save processed data to `data/processed/`

### 5. Run the API

Start the FastAPI backend:

```bash
uvicorn backend.app:app --reload
```

Then access:
- Interactive Docs: http://localhost:8000/docs
- Predict Mortality: http://localhost:8000/predict/mortality
- Health Check: http://localhost:8000/health

## Project Structure

```
famineSight/
├── backend/              # FastAPI backend
│   ├── app.py            # API entry point
│   ├── routers/          # API endpoints
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   └── main.py           # Application entry point
├── src/                  # Source code
│   ├── data/             # Data processing
│   │   ├── acled_fetcher.py  # ACLED API client
│   │   ├── preprocessor.py   # Full data pipeline
│   │   └── synthetic_generator.py # Synthetic data
│   ├── analysis/         # ML models
│   │   ├── classification.py # Supervised models
│   │   └── clustering.py     # Unsupervised clustering
│   ├── llm/              # AI models
│   │   ├── client.py       # Hybrid LLM client
│   │   ├── ollama_client.py  # Ollama integration
│   │   ├── groq_client.py    # Groq integration
│   │   └── prompt_templates.py # Prompt engineering
│   └── config.py         # Configuration
├── notebooks/            # Research notebooks
├── tests/                # Test suite
├── data/                 # Data directory
├── models/               # Trained models
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

## Configuration

Configure your system in `.env`:

```bash
# Enable Groq (default for better performance)
HAVE_GROQ_API_KEY=True

# Use local Ollama (default for offline use)
HAVE_OLLAMA=True

# Production mode
RUN_PRODUCTION=False
```

## Model Architecture

```mermaid
graph TD
    subgraph "Data Layer"
        IPC[IPC Data] --> Preprocessor[Data Preprocessor]
        ACLED[ACLED Data] --> Preprocessor
        TAMSAT[TAMSAT Data] --> Preprocessor
    end

    subgraph "Preprocessing"
        Preprocessor --> FeatureEngineering[Feature Engineering]
        FeatureEngineering --> MissingValueImputation[Missing Value Imputation]
        MissingValueImputation --> Scaling[Scaling]
    end

    subgraph "Modeling Layer"
        Scaling --> Clustering[Clustering]
        Scaling --> Classification[Classification]

        subgraph "Clustering"
            Clustering --> KMeans[K-Means]
            Clustering --> HDBSCAN[HDBSCAN]
        end

        subgraph "Classification"
            Classification --> RandomForest[Random Forest]
            Classification --> XGBoost[XGBoost]
        end
    end

    subgraph "LLM Layer"
        Classification --> HybridClient[Hybrid Client]
        HybridClient --> OllamaClient[Ollama]
        HybridClient --> GroqClient[Groq]
    end

    subgraph "Application Layer"
        HybridClient --> API[FastAPI Backend]
        API --> Streamlit[Streamlit Dashboard]
    end
```

## Production Deployment

### Docker Deployment

To build and run with Docker:

```bash
# Build the backend
docker build -t faminesight-backend ./backend

# Build the frontend
docker build -t faminesight-frontend ./frontend

# Run with Docker Compose
docker-compose up --build
```

Access the application:
- Frontend: http://localhost:8501
- API: http://localhost:8000

### Production Configuration

For production deployment, ensure:
- `RUN_PRODUCTION=True` in `.env`
- Real ACLED credentials are set
- Groq API key is configured
- Sufficient RAM (16GB+ recommended)

## Troubleshooting

### Common Issues

**1. LLM Connection Issues:**
```bash
# Check Ollama is running
docker ps
# Should see 'ollama/ollama' container
```

**2. ACLED Authentication Errors:**
```bash
# Verify credentials in .env
cat .env
# Check token generation
python src/data/acled_fetcher.py
```

**3. Missing Features:**
```bash
# Run full pipeline to generate all features
python src/data/preprocessor.py
```

### Production Readiness Score

```
Status: 🔴 NOT PRODUCTION-READY (35%/100)

Critical Bugs:
- 🔴 Model path mismatch - Backend will crash on startup
- 🔴 Feature dimension mismatch in API (7 vs 33 features)
- 🔴 Missing features in trained models
- 🔴 False negatives computed from wrong index
- 🔴 Temporal split data leakage

Missing Components:
- ❌ Frontend not implemented
- ❌ Train script not implemented
- ❌ Test suite not implemented
- ❌ Dockerfiles not implemented
- ❌ Comprehensive documentation
```

## AI Model Settings

| Component | Model | Purpose |
|-----------|-------|---------|
| **Clustering** | HDBSCAN (best), K-Means (fallback) | Unsupervised famine identification |
| **Classification** | Random Forest, XGBoost | Supervised predictive modeling |
| **Groq LLM** | llama-3.3-70b-versatile | Hybrid mode only (sentiment + anomaly detection) |
| **Ollama LLM** | qwen3:32b | Context-aware alerting, fallback for Groq |

## License

This project is for humanitarian purposes only. Unauthorized commercial use is strictly prohibited.

## Support