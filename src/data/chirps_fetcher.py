# src/data/chirps_fetcher.py
import requests
import pandas as pd
import logging
from pathlib import Path
from src.config import DATA_RAW, DISTRICT_PCODES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_chirps_data():
    """
    Fetch CHIRPS rainfall data for Somalia from HDX.
    Uses PCODE column directly (adm_level==2 districts) — no intermediate name mapping.
    """
    chirps_url = "https://data.humdata.org/dataset/ed6e1b4b-8094-47e6-bdf7-f6d56fa7abb9/resource/8b333d58-d69e-418c-b5e3-dd86f12eee05/download/som-rainfall-subnat-full.csv"

    try:
        logger.info(f"Downloading CHIRPS data from {chirps_url}...")
        response = requests.get(chirps_url, timeout=60)
        response.raise_for_status()

        raw_dir = DATA_RAW / "chirps"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "som_rainfall_subnat_raw.csv"

        with open(raw_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Saved raw CHIRPS data to {raw_path}")

        df = pd.read_csv(raw_path)
        logger.info(f"Loaded {len(df)} records from CHIRPS data")

        df['date'] = pd.to_datetime(df['date'])

        # Keep only admin2 districts with valid OCHA pcodes
        valid_pcodes = set(DISTRICT_PCODES.values())
        df = df[(df['adm_level'] == 2) & (df['PCODE'].isin(valid_pcodes))].copy()
        df.rename(columns={'PCODE': 'pcode'}, inplace=True)

        logger.info(f"After district filter: {len(df)} rows, {df['pcode'].nunique()} districts")

        df['month_start'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df = df.sort_values(['pcode', 'date'])

        # Take last dekad per month (gives end-of-month rolling total/anomaly)
        monthly_df = df.groupby(['pcode', 'month_start']).last().reset_index()

        monthly_df['rainfall'] = monthly_df['r1h']
        monthly_df['rainfall_anomaly_pct'] = monthly_df['r1q'] - 100

        final_df = monthly_df[['month_start', 'pcode', 'rainfall', 'rainfall_anomaly_pct']]
        final_df = final_df.rename(columns={'month_start': 'date'})

        output_path = raw_dir / "chirps_rainfall.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(
            f"Saved CHIRPS data to {output_path} ({len(final_df)} rows, "
            f"{final_df['pcode'].nunique()} districts, "
            f"{final_df['date'].min().date()} – {final_df['date'].max().date()})"
        )

        return final_df

    except Exception as e:
        logger.error(f"Error fetching CHIRPS data: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to fetch CHIRPS data
    """
    logger.info("Starting CHIRPS data fetch...")
    df = fetch_chirps_data()
    
    if not df.empty:
        logger.info("Successfully fetched and processed CHIRPS data")
    else:
        logger.error("CHIRPS data fetch failed")

if __name__ == "__main__":
    main()
