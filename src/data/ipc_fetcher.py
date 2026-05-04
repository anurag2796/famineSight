# src/data/ipc_fetcher.py
import requests
import pandas as pd
import logging
from pathlib import Path
from src.config import DATA_RAW, DISTRICT_PCODES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_ipc_data():
    """
    Fetch IPC food security phase data for Somalia from HDX
    """
    # Public CSV link for Somalia IPC historical area-level data (long format)
    ipc_url = "https://data.humdata.org/dataset/26cac16a-98cd-4c4e-9353-40bd423302c0/resource/80be59cd-6d1d-423f-9114-e2fb507fd257/download/ipc_som_area_long.csv"

    try:
        logger.info(f"Downloading IPC data from {ipc_url}...")
        response = requests.get(ipc_url, timeout=60)
        response.raise_for_status()

        # Save raw data
        raw_dir = DATA_RAW / "ipc"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "ipc_som_area_raw.csv"
        
        with open(raw_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Saved raw IPC data to {raw_path}")

        # Load data
        df = pd.read_csv(raw_path)
        logger.info(f"Loaded {len(df)} records from IPC data")

        # Process data
        # Standardize columns
        df = df.rename(columns={
            'From': 'date',
            'Area': 'area',
            'Phase': 'phase',
            'Percentage': 'percentage'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Map areas to internal names by checking if the OCHA district name is in the IPC area string
        def map_area(area_name):
            area_name_lower = str(area_name).lower()
            # Sort by length descending to match longest possible string first (e.g., 'Buur Hakaba' vs 'Buur')
            sorted_districts = sorted(DISTRICT_PCODES.keys(), key=len, reverse=True)
            for district in sorted_districts:
                if district.lower() in area_name_lower:
                    return district
            return None

        df['internal_name'] = df['area'].apply(map_area)
        
        # Filter to matched areas
        df = df.dropna(subset=['internal_name']).copy()
        
        # Map internal names to pcodes
        df['pcode'] = df['internal_name'].map(DISTRICT_PCODES)
        
        # Pivot phases to columns
        # We want ipc_phase0_pct, ipc_phase1_pct, etc.
        # Note: IPC levels in CSV are strings like '1', '2', '3', '4', '5', '3+'
        
        # Convert phase to a clean string format for pivoting
        df['phase_col'] = 'ipc_phase' + df['phase'].astype(str) + '_pct'
        
        # Aggregate to monthly level by pcode and phase
        # Note: IPC often has multiple rows per month for the same area (different phases)
        pivoted = df.pivot_table(
            index=['date', 'pcode', 'internal_name'],
            columns='phase_col',
            values='percentage',
            aggfunc='mean'
        ).reset_index()
        
        # Ensure all columns exist
        for i in range(1, 6):
            col = f'ipc_phase{i}_pct'
            if col not in pivoted.columns:
                pivoted[col] = 0.0
        
        # Rename columns to match preprocessor expected names
        # Preprocessor expects: ipc_phase4_pct, ipc_phase5_pct
        
        # Save processed data
        output_path = raw_dir / "ipc_phases.csv"
        pivoted.to_csv(output_path, index=False)
        logger.info(f"Saved processed IPC data to {output_path} ({len(pivoted)} rows)")

        return pivoted

    except Exception as e:
        logger.error(f"Error fetching IPC data: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to fetch IPC data
    """
    logger.info("Starting IPC data fetch...")
    df = fetch_ipc_data()
    
    if not df.empty:
        logger.info("Successfully fetched and processed IPC data")
    else:
        logger.error("IPC data fetch failed")

if __name__ == "__main__":
    main()
