# FamineSight - ACLED Data Access Setup

To enable real ACLED data access, please create a `.env` file in the project root with your ACLED API credentials:

```bash
# Create the .env file
echo "ACLED_API_KEY=your_actual_api_key_here" > .env
echo "ACLED_EMAIL=your_actual_email_here" >> .env
echo "OLLAMA_HOST=http://host.docker.internal:11434" >> .env
echo "OLLAMA_MODEL=qwen3:32b" >> .env
echo "RF_N_JOBS=4" >> .env
echo "XGB_DEVICE=cpu" >> .env
```

## Before Using Real ACLED Data

1. Obtain your ACLED API key from https://acleddata.com/
2. Register for an account if you haven't already
3. Once you have your credentials, replace the placeholder values in the `.env` file
4. Verify the credentials work by running:

```bash
python scripts/fetch_data.py
```

This will attempt to fetch real data. If credentials are correct, it will fetch real ACLED data; otherwise, it will fall back to synthetic data.

## Verification

After creating the `.env` file, you can verify it works by running:

```bash
python -c "from src.config import ACLED_API_KEY, ACLED_EMAIL; print('API Key set:', bool(ACLED_API_KEY)); print('Email set:', bool(ACLED_EMAIL))"
```

The system will automatically fall back to synthetic data if real data fetching fails, ensuring continuous operation.