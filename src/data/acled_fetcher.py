# src/data/acled_fetcher.py
import requests
import pandas as pd
import time
import logging
from pathlib import Path
from src.config import ACLED_EMAIL, ACLED_PASSWORD, ACLED_BASE_URL, DATA_RAW, DISTRICT_PCODES
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token storage
_token_cache = {}

# ---------------------------------------------------------------------------
# Build a comprehensive ACLED admin2 name → OCHA pcode lookup from DISTRICT_PCODES.
# DISTRICT_PCODES is now keyed by official OCHA district names (e.g. "Belet Weyne").
# We also add common ACLED spelling variants as aliases.
# ---------------------------------------------------------------------------
ACLED_ALIAS_MAP = {
    # ACLED spelling  →  OCHA district name (key in DISTRICT_PCODES)
    "Banadir":        "Bondhere",     # Banadir region → any Banadir sub-district, pick first
    "Kismaayo":       "Kismaayo",
    "Baydhaba":       "Baydhaba",
    "Hargeysa":       "Hargeysa",
    "Gaalkacyo":      "Gaalkacyo",
    "Belet Weyne":    "Belet Weyne",
    "Buur Hakaba":    "Buur Hakaba",
    "Laas Caanood":   "Laas Caanood",
    "Ceerigaabo":     "Ceerigaabo",
    "Burco":          "Burco",
    "Garoowe":        "Garoowe",
    "Bossaso":        "Bossaso",
    "Afgooye":        "Afgooye",
    "Luuq":           "Luuq",
    "Dhuusamarreeb":  "Dhuusamarreeb",
    "Jamaame":        "Jamaame",
    "Baardheere":     "Baardheere",
    "Doolow":         "Doolow",
    "Jowhar":         "Jowhar",
    "Marka":          "Marka",
    "Berbera":        "Berbera",
}

def get_acled_token():
    """
    Get ACLED OAuth token using email/password.

    Returns:
        Bearer token or None if authentication fails
    """
    # Check if we have a valid cached token
    if ACLED_EMAIL in _token_cache:
        # Simple check - in production, you'd want to check actual expiration
        return _token_cache[ACLED_EMAIL]

    if not ACLED_EMAIL or not ACLED_PASSWORD:
        logger.warning("ACLED_EMAIL or ACLED_PASSWORD not set. Cannot authenticate.")
        return None

    try:
        # OAuth token endpoint
        token_url = "https://acleddata.com/oauth/token"

        # Prepare authentication data
        auth_data = {
            "username": ACLED_EMAIL,
            "password": ACLED_PASSWORD,
            "grant_type": "password",
            "client_id": "acled",
            "scope": "authenticated"
        }

        # Make authentication request
        response = requests.post(token_url, data=auth_data, timeout=30)

        if response.status_code == 200:
            token_data = response.json()
            token = token_data.get("access_token")

            if token:
                _token_cache[ACLED_EMAIL] = token
                logger.info("Successfully obtained ACLED OAuth token")
                return token
            else:
                logger.error("No access_token in authentication response")
                return None
        else:
            logger.error(f"ACLED authentication failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error obtaining ACLED token: {str(e)}")
        return None

def fetch_acled_data(country="Somalia", start_date="2010-01-01", end_date="2024-12-31"):
    """
    Fetch ACLED data from the API with improved parameter handling.
    """
    if not ACLED_EMAIL or not ACLED_PASSWORD:
        logger.warning("ACLED API credentials not found. Returning empty DataFrame.")
        return pd.DataFrame()

    # Get authentication token
    token = get_acled_token()
    if not token:
        logger.warning("Unable to obtain ACLED authentication token. Returning empty DataFrame.")
        return pd.DataFrame()

    # Set up headers with authentication
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    all_data = []
    page = 1

    while True:
        try:
            # Prepare API parameters as per new documentation
            params = {
                'country': country,
                'event_date': f"{start_date}|{end_date}",
                'event_date_where': 'BETWEEN',
                'limit': 5000,
                'page': page,
                'fields': 'event_date|event_type|sub_event_type|actor1|admin1|admin2|latitude|longitude|fatalities|civilian_targeting'
            }

            # Make API request
            response = requests.get(ACLED_BASE_URL, headers=headers, params=params, timeout=60)

            # Handle rate limiting
            if response.status_code == 429:
                wait_time = min(60, 2 ** page)
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            if 'data' not in data or not data['data']:
                break

            page_data = data['data']
            all_data.extend(page_data)
            logger.info(f"Fetched page {page} with {len(page_data)} records (Total: {len(all_data)})")

            # Check if we got fewer than the limit, which means we're done
            if len(page_data) < 5000:
                break

            page += 1
            if page > 200: # Increased safety break
                logger.warning("Reached safety page limit, stopping")
                break

        except requests.exceptions.RequestException as e:
            logger.error(f"ACLED API request error on page {page}: {str(e)}")
            if "401" in str(e) or "Unauthorized" in str(e):
                if ACLED_EMAIL in _token_cache:
                    del _token_cache[ACLED_EMAIL]
                new_token = get_acled_token()
                if new_token:
                    headers["Authorization"] = f"Bearer {new_token}"
                    continue
            break
        except Exception as e:
            logger.error(f"Error fetching ACLED data: {str(e)}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = clean_acled_data(df)

    # Save raw data
    raw_path = DATA_RAW / "acled" / f"{country.lower()}_acled_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)

    return df

def clean_acled_data(df):
    """
    Clean and process ACLED data with robustness.
    """
    if df.empty:
        return df

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Select only the required fields
    required_fields = [
        'event_date', 'event_type', 'sub_event_type', 'actor1',
        'admin1', 'admin2', 'latitude', 'longitude',
        'fatalities', 'civilian_targeting'
    ]

    # Keep only existing fields
    existing_fields = [f for f in required_fields if f in df.columns]
    df = df[existing_fields]

    # Parse event_date
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

    # Cast fatalities to int safely
    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0).astype(int)

    # Extract civilian_targeting boolean safely
    if 'civilian_targeting' in df.columns:
        df['civilian_targeting'] = df['civilian_targeting'].fillna('No').astype(str).str.contains('Yes', case=False, na=False)
    else:
        df['civilian_targeting'] = False

    # Filter out invalid records
    df = df.dropna(subset=['event_date', 'admin2'])

    return df

def aggregate_acled_data(df):
    """
    Aggregate ACLED data to monthly district level.
    Matches admin2 names directly against DISTRICT_PCODES (OCHA names) or
    via ACLED_ALIAS_MAP for known spelling variants.
    """
    if df.empty:
        logger.warning("No data to aggregate")
        return df

    df = df.copy()

    # Step 1: try alias map for known ACLED-specific spelling variants
    df['ocha_name'] = df['admin2'].map(ACLED_ALIAS_MAP)
    # Step 2: if no alias, check if admin2 itself is a valid OCHA district name
    direct_match = df['ocha_name'].isna()
    df.loc[direct_match, 'ocha_name'] = df.loc[direct_match, 'admin2'].apply(
        lambda x: x if x in DISTRICT_PCODES else np.nan
    )
    # Step 3: fall back to admin1 alias
    still_missing = df['ocha_name'].isna()
    df.loc[still_missing, 'ocha_name'] = df.loc[still_missing, 'admin1'].map(ACLED_ALIAS_MAP)

    # Filter to matched districts only
    df = df.dropna(subset=['ocha_name'])
    if df.empty:
        logger.warning("No ACLED records matched to known OCHA districts")
        return df

    # Map OCHA name → OCHA pcode
    df['pcode'] = df['ocha_name'].map(DISTRICT_PCODES)
    df = df.dropna(subset=['pcode'])

    # Convert event_date to month start date
    df['date'] = df['event_date'].dt.to_period('M').dt.to_timestamp()

    # Group by pcode and month
    grouped = df.groupby(['pcode', 'ocha_name', 'date']).agg({
        'fatalities': 'sum',
        'civilian_targeting': 'sum',
        'event_type': 'count'
    }).reset_index()

    grouped = grouped.rename(columns={
        'ocha_name': 'district',
        'event_type': 'conflict_events',
        'fatalities': 'conflict_fatalities',
        'civilian_targeting': 'civilian_targeting_events'
    })

    # Save aggregated data
    output_path = DATA_RAW / "acled" / "somalia_acled.csv"
    grouped.to_csv(output_path, index=False)
    logger.info(
        f"Saved aggregated ACLED data → {output_path} "
        f"({len(grouped)} rows, {grouped['pcode'].nunique()} districts)"
    )

    return grouped

def main():
    """
    Main function to fetch and process ACLED data
    """
    logger.info("Starting ACLED data fetch...")

    # Create directory if it doesn't exist
    (DATA_RAW / "acled").mkdir(parents=True, exist_ok=True)

    # Fetch data
    df = fetch_acled_data()

    if not df.empty:
        logger.info("Successfully fetched ACLED data")
        # Aggregate data
        aggregated_df = aggregate_acled_data(df)
        return aggregated_df
    else:
        logger.warning("No ACLED data fetched")
        return pd.DataFrame()

if __name__ == "__main__":
    main()