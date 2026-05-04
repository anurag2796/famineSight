# src/data/fsnau_fetcher.py
import requests
import pandas as pd
import logging
from pathlib import Path
from src.config import DATA_RAW, DISTRICT_PCODES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from FSNAU Area names in Excel to internal names
# Based on common names in Somalia datasets
FSNAU_AREA_MAP = {
    "Mogadishu": "Mogadishu",
    "Kismayo": "Kismayo",
    "Baidoa": "Baidoa",
    "Afgooye": "Afgooye",
    "Luuq": "Luuq",
    "Hargeisa": "Hargeisa",
    "Berbera": "Berbera",
    "Galkayo": "Galkayo",
    "Dhuusamarreeb": "Dhuusamarreeb",
    "Jamaame": "Jamaame",
    "Buurhakaba": "Buurhakaba",
    "Bay": "Bay",
    "Gedo": "Gedo",
    "Mudug": "Mudug",
    "Sanaag": "Sanaag",
    "Togdheer": "Togdheer",
    "Sool": "Sool",
    "Galgaduud": "Galgaduud",
    "Hiiraan": "Hiiraan"
}

def fetch_fsnau_data():
    """
    Fetch FSNAU mortality data for Somalia from HDX (XLSX format)
    """
    # Public XLSX link for Somalia mortality estimation 2014-2018
    fsnau_url = "https://data.humdata.org/dataset/89d5e091-e4be-48f3-8ab1-bb169e5e0255/resource/e4da3c36-204c-476c-a666-a88afbd23633/download/2014-2018-somalia-death-rates.xlsx"

    try:
        logger.info(f"Downloading FSNAU data from {fsnau_url}...")
        response = requests.get(fsnau_url, timeout=60)
        response.raise_for_status()

        # Save raw data
        raw_dir = DATA_RAW / "fsnau"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "fsnau_mortality_raw.xlsx"
        
        with open(raw_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Saved raw FSNAU data to {raw_path}")

        # Load Excel data
        df = pd.read_excel(raw_path, sheet_name='Summary_data', engine='openpyxl')
        
        # Check for HXL tags in the first row of data
        if df.iloc[0].astype(str).str.contains('#').any():
            logger.info("HXL tags detected, skipping the first row...")
            df = df.iloc[1:].reset_index(drop=True)
            
        logger.info(f"Loaded {len(df)} records from FSNAU data")

        # Process data
        # Note: The structure of this XLSX needs to be standardized.
        # Based on HDX description, it has columns like 'Region', 'District', 'CDR', 'U5DR'
        # We'll try to find columns that look like these.
        
        # Standardize columns (lowercase)
        df.columns = [str(c).strip().lower() for c in df.columns]
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Try to find year column (look for 4-digit numbers or 'year' in name, but NOT 'death rate')
        year_col = None
        for col in df.columns:
            if 'year' in col and 'death rate' not in col:
                year_col = col
                break
        
        if year_col:
             # Convert to numeric and filter for valid years
             df['year_tmp'] = pd.to_numeric(df[year_col], errors='coerce')
             df = df.dropna(subset=['year_tmp']).copy()
             df['date'] = pd.to_datetime(df['year_tmp'].astype(int).astype(str) + '-01-01')
        else:
            # Fallback for the 2014-2018 dataset which lacks a year column but has historical averages
            logger.warning("No year column found, assigning 2016-01-01 (midpoint of 2014-2018)")
            df['date'] = pd.to_datetime('2016-01-01')

        # Try to find area/district column
        area_cols = [c for c in df.columns if 'district' in c or 'area' in c or 'region' in c]
        if not area_cols:
             logger.error("No area/district column found in FSNAU data")
             return pd.DataFrame()
             
        area_col = area_cols[0]
        
        # Map areas to internal names
        def map_area(area_name):
            if not isinstance(area_name, str):
                return None
            for key, val in FSNAU_AREA_MAP.items():
                if key.lower() in area_name.lower():
                    return val
            return None

        df['internal_name'] = df[area_col].apply(map_area)
        
        # Filter to matched areas
        df = df.dropna(subset=['internal_name']).copy()
        
        # Map internal names to pcodes
        df['pcode'] = df['internal_name'].map(DISTRICT_PCODES)
        
        # Identify CDR and U5DR columns
        cdr_cols = [c for c in df.columns if 'cdr' in c or 'crude' in c or 'death rate' in c]
        u5dr_cols = [c for c in df.columns if 'u5dr' in c or 'under' in c]
        
        if cdr_cols:
            df['cdr_per_10k_per_day'] = pd.to_numeric(df[cdr_cols[0]], errors='coerce')
        else:
            df['cdr_per_10k_per_day'] = 0.0
            
        if u5dr_cols:
            df['u5dr_per_10k_per_day'] = pd.to_numeric(df[u5dr_cols[0]], errors='coerce')
        else:
            df['u5dr_per_10k_per_day'] = 0.0
            
        # Clean NaNs in numeric columns
        df['cdr_per_10k_per_day'] = df['cdr_per_10k_per_day'].fillna(0.0)
        df['u5dr_per_10k_per_day'] = df['u5dr_per_10k_per_day'].fillna(0.0)
            
        # Define crisis label (1 if CDR > 1.0)
        df['crisis_label'] = (df['cdr_per_10k_per_day'] > 1.0).astype(int)
        
        # Keep only required columns
        final_df = df[['date', 'pcode', 'cdr_per_10k_per_day', 'u5dr_per_10k_per_day', 'crisis_label']]
        
        # Save processed data
        output_path = raw_dir / "fsnau_mortality.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed FSNAU data to {output_path} ({len(final_df)} rows)")

        return final_df

    except Exception as e:
        logger.error(f"Error fetching FSNAU data: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to fetch FSNAU data
    """
    logger.info("Starting FSNAU data fetch...")
    df = fetch_fsnau_data()
    
    if not df.empty:
        logger.info("Successfully fetched and processed FSNAU data")
    else:
        logger.error("FSNAU data fetch failed")

if __name__ == "__main__":
    main()
