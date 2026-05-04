#!/usr/bin/env python3
"""
Script to fetch all required data for FamineSight project.
This script tries real data sources first, falling back to synthetic data when needed.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.acled_fetcher import fetch_acled_data, aggregate_acled_data
from src.data.wfp_fetcher import fetch_wfp_data
from src.data.chirps_fetcher import fetch_chirps_data
from src.data.ipc_fetcher import fetch_ipc_data
from src.data.fsnau_fetcher import fetch_fsnau_data
from src.data.shapefile_fetcher import fetch_shapefile_data
from src.data.ndvi_fetcher import fetch_ndvi_data
from src.data.unhcr_fetcher import fetch_unhcr_data
from data.synthetic.generate_synthetic import (
    generate_chirps_data, generate_acled_data, 
    generate_wfp_data, generate_fsnau_data, generate_ipc_data
)
from src.config import DATA_RAW, DATA_SYNTHETIC

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_all_data(force_synthetic: bool = False):
    """
    Fetch all required data for the project.

    Args:
        force_synthetic: If True, skip real data fetching and use synthetic data only
    """
    logger.info("Starting data fetching process...")

    # Create data directories
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for sub in ["acled", "wfp", "fsnau", "ipc", "shapefiles", "chirps", "ndvi", "unhcr"]:
        (DATA_RAW / sub).mkdir(parents=True, exist_ok=True)

    # 0. Shapefile / district_lookup FIRST — other fetchers read DISTRICT_PCODES
    #    which is populated from district_lookup.csv at import time.
    logger.info("Fetching OCHA COD-AB Somalia administrative boundaries...")
    try:
        result = fetch_shapefile_data()
        if result:
            logger.info(
                f"[FETCHED REAL] Boundary data — "
                f"{result['admin2_features']} districts, "
                f"{result['admin1_features']} regions"
            )
        else:
            logger.warning("Boundary fetch returned empty — map visualisations will be unavailable")
    except Exception as e:
        logger.warning(f"Boundary fetch failed: {e} — map visualisations will be unavailable")

    # 1. Fetch ACLED data (extended to 2010)
    if force_synthetic:
        logger.info("[USING SYNTHETIC] ACLED data")
        generate_acled_data()
    else:
        logger.info("Fetching ACLED data (2010–2024)...")
        acled_df = fetch_acled_data(start_date="2010-01-01", end_date="2024-12-31")
        if not acled_df.empty:
            logger.info("[FETCHED REAL] ACLED data")
            # Aggregate to district level
            district_df = aggregate_acled_data(acled_df)
            district_df.to_csv(DATA_RAW / "acled" / "somalia_acled.csv", index=False)
            logger.info("Saved aggregated ACLED data to district level")
        else:
            logger.warning("Failed to fetch real ACLED data, using synthetic data")
            generate_acled_data()

    # 2. Fetch WFP data
    if force_synthetic:
        logger.info("[USING SYNTHETIC] WFP data")
        generate_wfp_data()
    else:
        logger.info("Fetching WFP data...")
        wfp_df = fetch_wfp_data()
        if not wfp_df.empty:
            logger.info("[FETCHED REAL] WFP data")
        else:
            logger.warning("Failed to fetch real WFP data, using synthetic data")
            generate_wfp_data()

    # 3. OCHA boundaries already fetched above (step 0)

    # 4. FSNAU Data
    try:
        logger.info("Fetching FSNAU data...")
        fsnau_df = fetch_fsnau_data()
        if not fsnau_df.empty:
            logger.info("[FETCHED REAL] FSNAU data")
        else:
            raise Exception("Real FSNAU fetch returned empty")
    except Exception as e:
        logger.warning(f"Real FSNAU fetch failed: {e}. Falling back to synthetic...")
        generate_fsnau_data()

    # 5. IPC Data
    try:
        logger.info("Fetching IPC data...")
        ipc_df = fetch_ipc_data()
        if not ipc_df.empty:
            logger.info("[FETCHED REAL] IPC data")
        else:
            raise Exception("Real IPC fetch returned empty")
    except Exception as e:
        logger.warning(f"Real IPC fetch failed: {e}. Falling back to synthetic...")
        generate_ipc_data()

    # 5. Fetch CHIRPS data (rainfall)
    if force_synthetic:
        logger.info("[USING SYNTHETIC] CHIRPS data")
        generate_chirps_data()
    else:
        logger.info("Fetching CHIRPS data...")
        chirps_df = fetch_chirps_data()
        if not chirps_df.empty:
            logger.info("[FETCHED REAL] CHIRPS data")
        else:
            logger.warning("Failed to fetch real CHIRPS data, using synthetic data")
            generate_chirps_data()

    # 7. NDVI (vegetation anomaly — MODIS)
    logger.info("Fetching NDVI data...")
    try:
        ndvi_df = fetch_ndvi_data()
        if not ndvi_df.empty:
            logger.info(f"[FETCHED REAL] NDVI data: {len(ndvi_df)} rows")
        else:
            logger.warning("NDVI fetch returned empty — ndvi_anomaly will be imputed")
    except Exception as e:
        logger.warning(f"NDVI fetch failed: {e} — ndvi_anomaly will be imputed")

    # 8. UNHCR displacement (IDPs + refugees)
    logger.info("Fetching UNHCR displacement data...")
    try:
        unhcr_df = fetch_unhcr_data()
        if not unhcr_df.empty:
            logger.info(f"[FETCHED REAL] UNHCR displacement data: {len(unhcr_df)} rows")
        else:
            logger.warning("UNHCR fetch returned empty — displacement features will be imputed")
    except Exception as e:
        logger.warning(f"UNHCR fetch failed: {e} — displacement features will be imputed")

    logger.info("Data fetching complete!")

if __name__ == "__main__":
    # Check if --synthetic flag is provided
    force_synthetic = "--synthetic" in sys.argv

    fetch_all_data(force_synthetic)