# FamineSight — ACLED Data Access Setup

FamineSight uses ACLED (Armed Conflict Location & Event Data) as its primary conflict data source. This document covers how to obtain credentials and configure the system to fetch real ACLED data.

## Step 1: Register with ACLED

1. Visit https://acleddata.com/ and create a free account.
2. Once registered, your **email** and **account password** serve as API credentials.
3. No separate API key is issued — authentication uses your email + password via OAuth 2.0.

## Step 2: Configure Your `.env` File

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Then add your ACLED credentials:

```bash
# ACLED Credentials
ACLED_EMAIL=your_email@example.com
ACLED_PASSWORD=your_acled_password

# LLM Configuration
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:32b
GROQ_API_KEY=                         # Optional — leave empty for Ollama-only mode
GROQ_MODEL=llama3-8b-8192

# API Security (required)
API_KEY=replace_with_a_strong_random_key

# Hardware tuning
RF_N_JOBS=4
XGB_DEVICE=cpu
```

Generate a secure `API_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Step 3: Verify the Connection

```bash
python -c "
from src.data.acled_fetcher import get_acled_token
token = get_acled_token()
print('Auth OK:', bool(token))
"
```

## Step 4: Fetch Data

```bash
# Fetch all data (will use ACLED credentials from .env)
python scripts/fetch_data.py

# Force synthetic data (no credentials required)
python scripts/fetch_data.py --synthetic
```

If credentials are correct, real ACLED data is downloaded to `data/raw/acled/`. If authentication fails, the system automatically falls back to synthetic data and continues operating normally.

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `401 Unauthorized` | Verify `ACLED_EMAIL` and `ACLED_PASSWORD` in `.env` |
| `429 Too Many Requests` | The fetcher has automatic exponential-backoff retry — wait and retry |
| `API_KEY not set` error | Generate a key and add `API_KEY=...` to `.env` |
| Module import errors | Ensure you're running from the project root with the venv activated |