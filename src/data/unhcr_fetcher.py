# src/data/unhcr_fetcher.py
"""
Fetch UNHCR displacement data (IDPs + refugees) for Somalia.

Sources (tried in order):
  1. UNHCR PRMN Somalia displacement XLSX (2016–present)
     https://data.humdata.org/dataset/somalia-internally-displaced-persons-idps
     Columns: Month End, Current (Arrival) District, Number of Individuals
  2. Synthesised national totals as last resort

Output:  data/raw/unhcr/unhcr_displacement_som.csv
         columns: date, pcode, idp_count, refugee_count
"""

import requests
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from src.config import DATA_RAW, DISTRICT_PCODES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = DATA_RAW / "unhcr"

# ---------------------------------------------------------------------------
# UNHCR Data API endpoints (public, no API key required)
# ---------------------------------------------------------------------------
UNHCR_API_BASE = "https://api.unhcr.org/population/v1"

# PRMN Somalia displacement XLSX (subnational, monthly, 2016–present)
PRMN_URL = (
    "https://data.humdata.org/dataset/475e2e3c-3cec-4961-b73c-d8e68791ce60"
    "/resource/981e0a25-8a83-48a2-a46c-0be81a881856"
    "/download/unhcr-prmn-displacement-dataset-1.xlsx"
)

# Admin1 region → representative OCHA admin2 pcode (for region-level fallback)
UNHCR_REGION_TO_PCODE = {
    "Banadir":          "SO2201",
    "Bay":              "SO2401",
    "Gedo":             "SO2601",
    "Hiraan":           "SO2001",
    "Hiraan ":          "SO2001",
    "Hiran":            "SO2001",
    "Lower Juba":       "SO2801",
    "Lower Shabelle":   "SO2301",
    "Middle Juba":      "SO2701",
    "Middle Shabelle":  "SO2101",
    "Mudug":            "SO1801",
    "Galgaduud":        "SO1901",
    "Sanaag":           "SO1501",
    "Togdheer":         "SO1301",
    "Sool":             "SO1401",
    "Woqooyi Galbeed":  "SO1201",
    "Awdal":            "SO1101",
    "Bakool":           "SO2501",
    "Nugaal":           "SO1701",
    "Bari":             "SO1601",
}


def _fetch_prmn() -> pd.DataFrame:
    """
    Download UNHCR PRMN Somalia displacement XLSX and parse into
    (date, pcode, idp_count). Covers 2016–present at admin2/monthly level.
    """
    try:
        logger.info(f"Downloading PRMN displacement XLSX: {PRMN_URL}")
        resp = requests.get(PRMN_URL, timeout=120, allow_redirects=True)
        resp.raise_for_status()
        raw_path = OUTPUT_DIR / "unhcr_prmn_raw.xlsx"
        with open(raw_path, "wb") as fh:
            fh.write(resp.content)
        logger.info(f"Downloaded {len(resp.content) / 1024:.0f} KB → {raw_path}")

        df = pd.read_excel(raw_path, sheet_name="OutputsArrivals")
        logger.info(f"PRMN raw shape: {df.shape}  cols: {list(df.columns)}")

        # Column names in PRMN XLSX
        date_col     = "Month End"
        district_col = "Current (Arrival) District"
        count_col    = "Number of Individuals"

        if not {date_col, district_col, count_col}.issubset(df.columns):
            logger.warning(f"Unexpected PRMN columns: {list(df.columns)}")
            return pd.DataFrame()

        df = df[[date_col, district_col, count_col]].copy()
        df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df["idp_count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int)

        # Map district names → OCHA pcodes
        df["pcode"] = df[district_col].map(DISTRICT_PCODES)
        df = df.dropna(subset=["date", "pcode"])

        result = (
            df.groupby(["date", "pcode"])["idp_count"]
            .sum()
            .reset_index()
        )
        result["refugee_count"] = 0
        logger.info(
            f"PRMN parsed: {len(result)} rows, {result['pcode'].nunique()} districts, "
            f"{result['date'].min().date()} – {result['date'].max().date()}"
        )
        return result

    except Exception as exc:
        logger.warning(f"PRMN fetch failed: {exc}")
        return pd.DataFrame()


def _build_synthetic_national_idp() -> pd.DataFrame:
    """
    Last-resort fallback: synthesise subnational IDP distribution from known
    national totals (UNHCR annual reports) distributed by population weight.

    Data source: UNHCR Global Trends 2010-2023 Somalia IDP figures (national).
    """
    logger.warning(
        "Using synthesised national IDP distribution as last resort. "
        "Run fetch_data.py again once connectivity is restored for real data."
    )

    # Known national IDP totals from UNHCR Global Trends reports (thousands)
    NATIONAL_IDP_K = {
        2010: 1460, 2011: 1460, 2012: 1136, 2013: 1070, 2014: 1106,
        2015: 1106, 2016: 1107, 2017: 2650, 2018: 2645, 2019: 2618,
        2020: 2988, 2021: 2969, 2022: 3776, 2023: 3864, 2024: 3600,
    }

    # Relative IDP burden per OCHA admin1 region (from IOM DTM baseline ~2022)
    REGION_WEIGHT = {
        "SO2201": 0.18,  # Banadir
        "SO2401": 0.12,  # Bay
        "SO2301": 0.10,  # Lower Shabelle
        "SO2601": 0.07,  # Gedo
        "SO2001": 0.07,  # Hiraan
        "SO1801": 0.06,  # Mudug
        "SO1901": 0.05,  # Galgaduud
        "SO2801": 0.05,  # Lower Juba
        "SO2501": 0.04,  # Bakool
        "SO2701": 0.04,  # Middle Juba
        "SO2101": 0.04,  # Middle Shabelle
        "SO1701": 0.03,  # Nugaal
        "SO1601": 0.03,  # Bari
        "SO1101": 0.02,  # Awdal
        "SO1201": 0.02,  # Woqooyi Galbeed
        "SO1301": 0.02,  # Togdheer
        "SO1401": 0.02,  # Sool
        "SO1501": 0.02,  # Sanaag
        "SO2401": 0.01,  # Bay (secondary entry)
        "SO1401": 0.01,  # Sool (secondary)
    }

    rows = []
    for year, total_k in NATIONAL_IDP_K.items():
        for month in range(1, 13):
            date = pd.Timestamp(year=year, month=month, day=1)
            for pcode, weight in REGION_WEIGHT.items():
                if pcode in set(DISTRICT_PCODES.values()):
                    rows.append({
                        "date": date,
                        "pcode": pcode,
                        "idp_count": int(total_k * 1000 * weight),
                        "refugee_count": 0,
                    })

    return pd.DataFrame(rows)


def fetch_unhcr_data() -> pd.DataFrame:
    """
    Main entry point. Returns DataFrame with: date, pcode, idp_count, refugee_count
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = _fetch_prmn()
    if not result.empty:
        out_path = OUTPUT_DIR / "unhcr_displacement_som.csv"
        result.to_csv(out_path, index=False)
        logger.info(f"Saved UNHCR displacement data → {out_path} ({len(result)} rows)")
        return result

    logger.warning("PRMN download failed — using synthesised national IDP distribution")
    result = _build_synthetic_national_idp()
    out_path = OUTPUT_DIR / "unhcr_displacement_som.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Saved synthesised UNHCR data → {out_path} ({len(result)} rows)")
    return result


def main():
    logger.info("Starting UNHCR displacement data fetch...")
    df = fetch_unhcr_data()
    if not df.empty:
        logger.info(f"UNHCR fetch complete: {df.shape}")
    else:
        logger.error("UNHCR fetch returned empty DataFrame")


if __name__ == "__main__":
    main()
