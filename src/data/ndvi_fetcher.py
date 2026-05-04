# src/data/ndvi_fetcher.py
"""
Fetch NDVI (vegetation health) data for Somalia from HDX.

Source:  https://data.humdata.org/dataset/som-ndvi-subnational
Dataset: Subnational NDVI time-series (dekadal), admin1 + admin2 combined.
         Columns: date, adm_level, adm_id, PCODE, n_pixels, vim, vim_avg, viq
         vim = current NDVI (0–1 scale); vim_avg = long-run mean NDVI (0–1).

We filter to adm_level==2 (districts) and compute:
  ndvi_anomaly = (vim - vim_avg) / vim_avg * 100
so negative = below-average vegetation (drought / degradation signal).

Output:  data/raw/ndvi/ndvi_somalia.csv  with columns: date, pcode, ndvi_anomaly
"""

import requests
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.config import DATA_RAW, DISTRICT_PCODES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HDX dataset URLs — eMODIS NDVI Somalia (public, no authentication required)
# Primary: subnational CSV per HDX; secondary: alternative resource ID.
# ---------------------------------------------------------------------------
NDVI_URLS = [
    # Full time-series (2002–present), admin1+admin2
    "https://data.humdata.org/dataset/f1e50c5b-304e-4e42-862b-cdc3d9016014/resource/3ecf339e-c53c-4726-8172-9a777baa5857/download/som-ndvi-subnat-full.csv",
    # 5-year rolling window (smaller download, same schema)
    "https://data.humdata.org/dataset/f1e50c5b-304e-4e42-862b-cdc3d9016014/resource/a1198654-9b82-4035-a00b-13384f45a28e/download/som-ndvi-subnat-5ytd.csv",
]

OUTPUT_DIR = DATA_RAW / "ndvi"

# CHIRPS-style PCODE → OCHA admin2 name map (for back-mapping if needed)
# The NDVI CSV uses standard OCHA admin2 pcodes directly, so we just filter
# against DISTRICT_PCODES values.

def fetch_ndvi_data() -> pd.DataFrame:
    """
    Download and process MODIS NDVI data for Somalia.

    Returns a DataFrame with columns: date, pcode, ndvi_anomaly
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / "ndvi_somalia_raw.csv"

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    content = None
    for url in NDVI_URLS:
        try:
            logger.info(f"Trying NDVI URL: {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            content = resp.content
            logger.info(f"Downloaded {len(content) / 1024:.1f} KB from {url}")
            break
        except Exception as exc:
            logger.warning(f"NDVI URL failed ({url}): {exc}")

    if content is None:
        logger.error("All NDVI download URLs failed. Returning empty DataFrame.")
        return pd.DataFrame()

    with open(raw_path, "wb") as fh:
        fh.write(content)
    logger.info(f"Saved raw NDVI data → {raw_path}")

    # ------------------------------------------------------------------
    # 2. Parse
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(raw_path, encoding="utf-8", low_memory=False)
    except Exception as exc:
        logger.error(f"Failed to parse NDVI CSV: {exc}")
        return pd.DataFrame()

    logger.info(f"Raw NDVI shape: {df.shape}  columns: {list(df.columns)}")

    required = {"date", "adm_level", "PCODE", "vim", "vim_avg"}
    if not required.issubset(df.columns):
        logger.error(f"Unexpected columns. Expected {required}, got {list(df.columns)}")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Filter to district level (adm_level == 2) and clean
    # ------------------------------------------------------------------
    df = df[df["adm_level"] == 2][["date", "PCODE", "vim", "vim_avg"]].copy()
    df.rename(columns={"PCODE": "pcode"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["vim"] = pd.to_numeric(df["vim"], errors="coerce")
    df["vim_avg"] = pd.to_numeric(df["vim_avg"], errors="coerce")
    df = df.dropna(subset=["date", "pcode", "vim", "vim_avg"])

    # vim is 0–1 scale; mask fill/water values
    df = df[(df["vim"] >= 0) & (df["vim"] <= 1) & (df["vim_avg"] > 0)]

    # Filter to known district pcodes
    valid_pcodes = set(DISTRICT_PCODES.values())
    df = df[df["pcode"].isin(valid_pcodes)]

    if df.empty:
        logger.warning("No NDVI records matched to known OCHA district pcodes.")
        return pd.DataFrame()

    # Aggregate to monthly (data is dekadal — 3 obs per month)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby(["pcode", "month"])[["vim", "vim_avg"]]
        .mean()
        .reset_index()
        .rename(columns={"month": "date"})
    )

    # ------------------------------------------------------------------
    # 4. Compute anomaly: (vim - vim_avg) / vim_avg * 100
    #    vim_avg is the long-run mean already provided by the dataset.
    # ------------------------------------------------------------------
    monthly["ndvi_anomaly"] = (
        (monthly["vim"] - monthly["vim_avg"]) / monthly["vim_avg"] * 100
    ).round(2)

    final = monthly[["date", "pcode", "ndvi_anomaly"]]

    # Save processed
    out_path = OUTPUT_DIR / "ndvi_somalia.csv"
    final.to_csv(out_path, index=False)
    logger.info(
        f"Saved NDVI data → {out_path} "
        f"({len(final)} rows, {final['pcode'].nunique()} districts, "
        f"{final['date'].min().date()} – {final['date'].max().date()})"
    )
    return final


def main():
    logger.info("Starting NDVI data fetch...")
    df = fetch_ndvi_data()
    if not df.empty:
        logger.info(f"NDVI fetch complete: {df.shape}")
    else:
        logger.error("NDVI fetch failed or returned empty")


if __name__ == "__main__":
    main()
