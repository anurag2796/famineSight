# Deployment Guide

## Overview

This document provides instructions for deploying FamineSight in development and production environments, including Jetson AGX Orin platforms and standard x86 Linux systems.

## System Requirements

### Minimum (x86 Development)
- **CPU**: Any modern x86-64 processor
- **Memory**: 16 GB RAM
- **Storage**: 20 GB SSD
- **OS**: Ubuntu 20.04+ or any Linux distro
- **Software**: Python 3.11+, Docker & Docker Compose v2

### Recommended (Jetson AGX Orin Production)
- **CPU**: ARM64 — Jetson AGX Orin 64 GB
- **Memory**: 60+ GB unified RAM (shared with GPU)
- **Storage**: 100 GB NVMe SSD
- **OS**: Ubuntu 22.04 (JetPack 6.x)
- **Software**: Docker Engine, Docker Compose v2, Ollama (on host)

## Environment Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd famineSight
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set all required values:

```bash
# Required
ACLED_EMAIL=your_email@organization.org
ACLED_PASSWORD=your_acled_password
API_KEY=<generate_with_command_below>

# LLM (Ollama is the production default; Groq is optional)
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:32b
GROQ_API_KEY=          # leave empty for offline/production
GROQ_MODEL=llama3-8b-8192

# CORS — list all trusted frontend origins
ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501,http://frontend:8501

# Hardware tuning (critical for Jetson)
RF_N_JOBS=4
XGB_DEVICE=cpu
```

Generate a secure `API_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Set secure file permissions:
```bash
chmod 600 .env
```

### 3. Install System Dependencies (Jetson / ARM64 only)

```bash
# GDAL is required for shapefile parsing
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev libgeos-dev libproj-dev build-essential
```

### 4. Prepare Data and Models

```bash
# Install Python deps (for scripting outside Docker)
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Fetch all data sources
python scripts/fetch_data.py

# Train ML models
python scripts/train_pipeline.py
```

## Docker Deployment

### Development Mode

```bash
# Build and start all services (backend + frontend + Ollama sidecar)
docker compose -f docker-compose.dev.yml up --build

# Run in background
docker compose -f docker-compose.dev.yml up -d

# Check service status
docker compose -f docker-compose.dev.yml ps
```

Resource limits in `docker-compose.dev.yml`:
- Backend: **4 GB** memory
- Frontend: **1 GB** memory
- Ollama sidecar: **2 GB** memory (for small dev model)

### Verify Services

```bash
# Backend health check
curl -H "X-API-Key: your_api_key" http://localhost:8000/health

# Frontend
open http://localhost:8501

# API docs
open http://localhost:8000/docs
```

### Build Individual Services

```bash
docker compose -f docker-compose.dev.yml build backend
docker compose -f docker-compose.dev.yml build frontend
```

## Local Development (Without Docker)

```bash
# Terminal 1: Backend
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
source venv/bin/activate
streamlit run frontend/app.py --server.port 8501
```

## Production Configuration

### Ollama Setup (Host Machine)

FamineSight requires Ollama running on the **host machine** (not in Docker), accessible via `host.docker.internal:11434`:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the production model
ollama pull qwen3:32b

# Verify
ollama list
curl http://localhost:11434/api/tags
```

### Resource Limits for Production

Adjust Docker memory limits based on your hardware:

```yaml
# docker-compose.dev.yml (or your production compose file)
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 8G    # Increase for larger datasets
          cpus: "4.0"
  frontend:
    deploy:
      resources:
        limits:
          memory: 2G
```

### Security Hardening

```bash
# Strict .env permissions
chmod 600 .env

# Verify no credentials in code
git log --all --oneline | xargs git show --stat | grep -i "password\|api_key\|secret"

# Restrict CORS to your actual frontend URL
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

## Data Management

### Verify Data Was Fetched

```bash
ls data/raw/acled/
ls data/raw/wfp/
ls data/raw/chirps/
ls data/raw/ndvi/
ls data/raw/unhcr/
ls data/raw/ipc/
ls data/raw/fsnau/
ls data/raw/shapefiles/
```

### Re-fetch Data

```bash
# Refetch all sources
python scripts/fetch_data.py

# Force synthetic (no credentials required)
python scripts/fetch_data.py --synthetic
```

### Backup

```bash
# Backup processed data and models
tar -czf backup-data-$(date +%Y%m%d).tar.gz data/processed/
tar -czf backup-models-$(date +%Y%m%d).tar.gz models/

# Backup environment config
cp .env .env.backup-$(date +%Y%m%d)
```

## Monitoring and Logging

### Service Logs

```bash
# Tail all logs
docker compose -f docker-compose.dev.yml logs -f

# Specific service
docker compose -f docker-compose.dev.yml logs -f backend
docker compose -f docker-compose.dev.yml logs -f frontend
```

### Health Checks

```bash
# Backend health (includes model load status + Ollama reachability)
curl -H "X-API-Key: your_api_key" http://localhost:8000/health

# System resources
htop
docker stats
```

## Maintenance

### Update Code

```bash
git pull origin main
docker compose -f docker-compose.dev.yml build --no-cache
docker compose -f docker-compose.dev.yml up -d
```

### Update Ollama Model

```bash
ollama pull qwen3:32b
docker compose -f docker-compose.dev.yml restart backend
```

### Retrain Models

```bash
# Fetch fresh data
python scripts/fetch_data.py

# Retrain
python scripts/train_pipeline.py

# Restart backend to reload model artifacts
docker compose -f docker-compose.dev.yml restart backend
```

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `API_KEY not set` | Add `API_KEY=...` to `.env` |
| Backend fails to start | Check `docker compose logs backend`; verify all env vars |
| `401` on API calls | Include `-H "X-API-Key: your_api_key"` header |
| Ollama not reachable | Run `ollama list` on host; ensure port 11434 is open |
| OOM on Jetson | Ensure `RF_N_JOBS=4` and `XGB_DEVICE=cpu` |
| Docker build GDAL error | Install system GDAL: `sudo apt-get install libgdal-dev` |
| Models not found | Run `python scripts/train_pipeline.py` first |

### Recovery

```bash
# Full restart
docker compose -f docker-compose.dev.yml down
docker compose -f docker-compose.dev.yml up -d

# Data pipeline reset (re-fetch + retrain)
python scripts/fetch_data.py --synthetic
python scripts/train_pipeline.py
```

## Conclusion

FamineSight is designed for robust deployment on both standard Linux systems and ARM64 Jetson platforms. The key differences for Jetson are:
- `RF_N_JOBS=4` and `XGB_DEVICE=cpu` are mandatory
- GDAL system libraries must be pre-installed
- Ollama must run on the host with sufficient RAM for `qwen3:32b` (≈20 GB)

Regular data refreshes (`fetch_data.py`), model retraining, and log monitoring are essential for sustained production operation.