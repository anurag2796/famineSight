# src/data/wfp_fetcher.py
import requests
import pandas as pd
import logging
from pathlib import Path
from src.config import DATA_RAW, DISTRICT_PCODES
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WFP uses admin1 region names that differ from OCHA admin2 district names.
# Map WFP admin1 → representative OCHA admin2 pcode for that region.
# When admin2 is available and matches an OCHA district, that takes priority.
# ---------------------------------------------------------------------------
WFP_ADMIN1_TO_PCODE = {
    # WFP admin1 name       → OCHA admin2 pcode (most populated / representative district)
    "Awdal":                "SO1101",   # Borama
    "Bakool":               "SO2501",   # Xudur
    "Banadir":              "SO2201",   # Bondhere (Mogadishu)
    "Bari":                 "SO1601",   # Bossaso
    "Bay":                  "SO2401",   # Baydhaba (Baidoa)
    "Galgaduud":            "SO1901",   # Dhuusamarreeb
    "Gedo":                 "SO2601",   # Garbahaarey
    "Hiraan":               "SO2001",   # Belet Weyne
    "Lower Juba":           "SO2801",   # Kismaayo
    "Lower Shabelle":       "SO2301",   # Marka
    "Middle Juba":          "SO2701",   # Bu'aale
    "Middle Shabelle":      "SO2101",   # Jowhar
    "Mudug":                "SO1801",   # Gaalkacyo
    "Nugaal":               "SO1701",   # Garoowe
    "Sanaag":               "SO1501",   # Ceerigaabo
    "Sool":                 "SO1401",   # Laas Caanood
    "Togdheer":             "SO1301",   # Burco
    "Woqooyi Galbeed":      "SO1201",   # Hargeysa
    # Common HDX variant spellings
    "Hiran":                "SO2001",
    "Hiraan ":              "SO2001",
    "Lower Jubba":          "SO2801",
    "Middle Jubba":         "SO2701",
    "Lower Shabele":        "SO2301",
    "Middle Shabele":       "SO2101",
    "Woqooyi Galbeed ":     "SO1201",
}

def fetch_wfp_data():
    """
    Fetch WFP food prices data for Somalia from HDX.
    Maps admin regions to OCHA admin2 pcodes for consistency.
    """
    # Correct WFP Somalia food prices URL from HDX
    wfp_url = "https://data.humdata.org/dataset/26727d1b-af49-4323-9215-c2ac479abb87/resource/39614bfb-0f9c-4800-8997-e68e41a38ced/download/wfp_food_prices_som.csv"

    try:
        # Make the request
        response = requests.get(wfp_url, timeout=30)
        response.raise_for_status()

        # Save raw data
        output_path = DATA_RAW / "wfp" / "wfp_prices_som_raw.csv"
        with open(output_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Saved raw WFP data to {output_path}")

        # Read the CSV file
        df = pd.read_csv(output_path)
        logger.info(f"Raw WFP data shape: {df.shape}")

        # Process the data
        df['date'] = pd.to_datetime(df['date'])

        # Priority 1: exact admin2 name match against OCHA district names
        df['pcode'] = df['admin2'].map(DISTRICT_PCODES)
        # Priority 2: admin1 region → representative district pcode
        missing = df['pcode'].isna()
        df.loc[missing, 'pcode'] = df.loc[missing, 'admin1'].map(WFP_ADMIN1_TO_PCODE)

        # Drop rows with no pcode mapping
        df = df.dropna(subset=['pcode'])

        # Create a simple price index based on average USD price of cereals per month
        cereals = df[df['category'].str.contains('cereals', case=False, na=False)]

        if not cereals.empty:
            # Group by date and pcode
            monthly_prices = cereals.groupby(
                [pd.Grouper(key='date', freq='MS'), 'pcode']
            )['usdprice'].mean().reset_index()

            # Simple index: normalize to median price = 100
            median_price = monthly_prices['usdprice'].median()
            monthly_prices['food_price_index'] = (monthly_prices['usdprice'] / median_price) * 100

            final_df = monthly_prices[['date', 'pcode', 'food_price_index']]
            logger.info(
                f"WFP price index: {len(final_df)} monthly rows, "
                f"{final_df['pcode'].nunique()} districts"
            )
        else:
            logger.warning("Could not calculate price index")
            final_df = pd.DataFrame(columns=['date', 'pcode', 'food_price_index'])

        # Save processed data
        processed_path = DATA_RAW / "wfp" / "wfp_prices_som.csv"
        final_df.to_csv(processed_path, index=False)
        logger.info(f"Saved processed WFP data to {processed_path}")

        return final_df

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching WFP data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error processing WFP data: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to fetch WFP data
    """
    logger.info("Starting WFP data fetch...")

    # Create directory if it doesn't exist
    (DATA_RAW / "wfp").mkdir(parents=True, exist_ok=True)

    # Fetch data
    df = fetch_wfp_data()

    if not df.empty:
        logger.info("WFP data fetch complete")
        return df
    else:
        logger.warning("No WFP data fetched")
        return pd.DataFrame()

if __name__ == "__main__":
    main()