# ACLED Integration Guide

## Overview

FamineSight integrates with the ACLED (Armed Conflict Location and Event Data) API to obtain conflict data for Somalia. ACLED data covers conflict events, fatalities, and civilian targeting events from 2010 to present, aggregated at the district-month level for use in the prediction pipeline.

## Authentication

### Credentials

ACLED uses **OAuth 2.0 with Password Grant**. Authentication is based on your ACLED account email and password — no separate API key is issued.

Configure in `.env`:
```bash
ACLED_EMAIL=your_email@organization.org
ACLED_PASSWORD=your_acled_password
```

### OAuth 2.0 Flow

1. **Token Retrieval** — POST to `https://acleddata.com/oauth/token`:
   ```http
   POST https://acleddata.com/oauth/token
   Content-Type: application/x-www-form-urlencoded

   username=your_email&password=your_password&grant_type=password&client_id=acled&scope=authenticated
   ```

2. **Token Usage** — Include in the `Authorization` header:
   ```http
   GET https://acleddata.com/api/acled/read?country=Somalia&year=2010:2024
   Authorization: Bearer <token>
   ```

3. **Token Lifetime** — Valid for 24 hours; automatically refreshed on `401` responses.

## Data Fetched

### API Endpoint
- **Base URL:** `https://acleddata.com/api/acled/read`
- **Country:** Somalia
- **Date range:** `2010-01-01` to `2024-12-31` (configurable via `DATA_START_DATE` / `DATA_END_DATE` in `config.py`)
- **Pagination:** 500 records per page, automatically paginated

### Response Fields Used

| Field | Type | Description |
|-------|------|-------------|
| `event_date` | string | Event date (parsed to datetime) |
| `event_type` | string | Type of conflict event |
| `admin1` | string | Region |
| `admin2` | string | District |
| `latitude` | float | Event latitude |
| `longitude` | float | Event longitude |
| `fatalities` | int | Number of fatalities |
| `civilian_targeting` | string | Civilian targeting flag |

### Aggregated Output

The fetcher aggregates to **district × month** level, producing:
- `conflict_events` — Total event count
- `conflict_fatalities` — Total fatalities
- `civilian_targeting_events` — Events explicitly targeting civilians

Raw output saved to: `data/raw/acled/somalia_acled_raw.csv`

## Implementation Details

### `src/data/acled_fetcher.py`

Key functions:
- `get_acled_token()` — Retrieves OAuth token; caches in memory; refreshes on expiry
- `fetch_acled_data()` — Paginates through all results with exponential-backoff retry
- `process_acled_data(df)` — Cleans, filters, and aggregates to district-month

### Error Handling

| HTTP Status | Behavior |
|-------------|---------|
| `401 Unauthorized` | Refreshes token and retries |
| `429 Too Many Requests` | Exponential backoff: `2^page` seconds, max 5 retries |
| `5xx Server Errors` | Retry with backoff |
| Any failure | Falls back to synthetic conflict data |

### Testing Authentication

```bash
python -c "
from src.data.acled_fetcher import get_acled_token
token = get_acled_token()
print('Auth OK:', bool(token))
print('Token length:', len(token) if token else 0)
"
```

Or run the full fetch:
```bash
python scripts/fetch_data.py
```

## Manual Download (Fallback)

If API access is unavailable:

1. Visit https://acleddata.com/data-export-tool/
2. Filter: **Region** → Africa → **Country** → Somalia
3. **Date range:** January 1, 2010 → Present
4. Leave **Event Types** blank (all types)
5. Export as CSV
6. Save to: `data/raw/acled/somalia_acled_raw.csv`

The preprocessor will pick up the manual CSV automatically.

## Security

- Credentials stored exclusively in `.env` (never in code)
- Set `.env` permissions: `chmod 600 .env`
- Tokens are held in memory only — never written to disk
- All ACLED API communication is over HTTPS

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `401 Unauthorized` | Verify `ACLED_EMAIL` and `ACLED_PASSWORD` in `.env` |
| `429 Too Many Requests` | Automatic backoff is implemented; wait and retry |
| Empty CSV output | Check date range; verify account has API access enabled |
| Token not refreshing | Restart the script; token cache is in-memory per process |

```bash
# Debug: test token retrieval
python -c "from src.data.acled_fetcher import get_acled_token; print(get_acled_token())"

# Debug: check env vars
grep ACLED .env
```