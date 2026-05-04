# FamineSight Data Sourcing Guide

This document provides exact instructions on how to obtain the raw data required for the FamineSight prediction models, either through manual CSV downloads or via the automated fetch scripts.

> **TL;DR:** Run `python scripts/fetch_data.py` — most sources are fetched automatically. Only ACLED requires credentials.

---

## 1. ACLED (Armed Conflict Location & Event Data)
**Target Metrics:** Conflict events, fatalities, civilian targeting events

**Credentials Required:** Yes (free registration)

- **Register:** https://acleddata.com/
- **Credentials:** Your account email + password (no separate API key)
- **Configure in `.env`:**
  ```bash
  ACLED_EMAIL=your_email@example.com
  ACLED_PASSWORD=your_acled_password
  ```
- **Auto-fetched by:** `src/data/acled_fetcher.py`
- **Save path:** `data/raw/acled/somalia_acled_raw.csv`
- **Manual download:** [ACLED Data Export Tool](https://acleddata.com/data-export-tool/)
  - Filter: Region → Africa → Country → Somalia
  - Date range: January 1, 2010 to Present
  - Export all event types

---

## 2. IPC (Integrated Food Security Phase Classification)
**Target Metrics:** AFI Phase 1–5 population percentages

**Credentials Required:** No (auto-fetched)

- **Auto-fetched by:** `src/data/ipc_fetcher.py`
- **Save path:** `data/raw/ipc/ipc_phases.csv`
- **Manual download:** [IPC Global Platform — Somalia](https://www.ipcinfo.org/ipc-country-analysis/)
  - Navigate to the Somalia country page
  - Select **Acute Food Insecurity (AFI)** analysis (not Malnutrition)
  - Click "Download Data" / "Export CSV"
  - Ensure the download covers **January 2010 to Present**

---

## 3. WFP (World Food Programme) Food Prices
**Target Metrics:** Market food price indices for Somalia

**Credentials Required:** No (auto-fetched via HDX)

- **Auto-fetched by:** `src/data/wfp_fetcher.py`
- **Save path:** `data/raw/wfp/wfp_prices_som_raw.csv`
- **Manual download:** [HDX WFP Food Prices — Somalia](https://data.humdata.org/dataset/26727d1b-af49-4323-9215-c2ac479abb87)
  - Download the CSV resource `wfp_food_prices_som.csv`
  - Verify it covers at least 2010 to present

---

## 4. CHIRPS (Climate Hazards Group InfraRed Precipitation)
**Target Metrics:** Monthly rainfall (mm) by district

**Credentials Required:** No (auto-fetched via HDX)

- **Auto-fetched by:** `src/data/chirps_fetcher.py`
- **Save path:** `data/raw/chirps/chirps_rainfall.csv`
- **Manual download:** [HDX Somalia Climate Data](https://data.humdata.org/dataset/somalia-climate) or [UC Santa Barbara CHIRPS Archive](https://data.chc.ucsb.edu/products/CHIRPS-2.0/)
  - Download district-level Somalia precipitation CSVs spanning 2010 to present

---

## 5. NDVI (Normalized Difference Vegetation Index)
**Target Metrics:** Vegetation health / drought proxy by district

**Credentials Required:** No (auto-fetched via HDX)

- **Auto-fetched by:** `src/data/ndvi_fetcher.py`
- **Save path:** `data/raw/ndvi/ndvi_somalia.csv`
- **Manual download:** [HDX Somalia NDVI Data](https://data.humdata.org/dataset/somalia-ndvi)

---

## 6. UNHCR Displacement Data
**Target Metrics:** Internally displaced persons (IDP) count, refugee count

**Credentials Required:** No (auto-fetched via UNHCR API)

- **Auto-fetched by:** `src/data/unhcr_fetcher.py`
- **Save path:** `data/raw/unhcr/unhcr_displacement.csv`
- **Manual download:** [UNHCR Somalia Displacement Data](https://www.unhcr.org/operational/situations/somalia-situation/)

---

## 7. OCHA COD-AB District Shapefiles
**Target Metrics:** Somalia admin2 district boundaries and p-codes

**Credentials Required:** No (auto-fetched via HDX)

- **Auto-fetched by:** `src/data/shapefile_fetcher.py`
- **Save path:** `data/raw/shapefiles/`
  - `district_lookup.csv` — district name → OCHA p-code mapping (92 districts)
  - `somalia_admin2.*` — shapefile for the vulnerability map
- **Manual download:** [OCHA COD-AB Somalia Admin2](https://data.humdata.org/dataset/cod-ab-som)

---

## 8. FSNAU (Food Security and Nutrition Analysis Unit)
**Target Metrics:** Crude Death Rate (CDR), Under-5 Death Rate (U5DR)

**Credentials Required:** Institutional access or manual PDF extraction

- **Auto-fetched by:** `src/data/fsnau_fetcher.py` (generates synthetic estimates when real data is unavailable)
- **Save path:** `data/raw/fsnau/fsnau_mortality.csv`
- **Manual access:** Data must be requested directly from FAO/FSNAU via their data-sharing agreements, or extracted manually from tables in their PDF reports at [fsnau.org](https://fsnau.org/). Request longitudinal data covering 2010 to present.
  - If real FSNAU data is unavailable, `fsnau_fetcher.py` generates synthetic mortality estimates — the pipeline remains fully functional.

---

## Running the Automated Fetch

Once `.env` is configured:

```bash
# Fetch all sources (ACLED uses real credentials; others are automatic)
python scripts/fetch_data.py

# Skip ACLED and use synthetic data for everything
python scripts/fetch_data.py --synthetic
```

All raw files are saved to `data/raw/<source>/` and processed output goes to `data/processed/merged_data.csv`.

---

## Data Coverage

| Source | Date Range | Granularity |
|--------|-----------|-------------|
| ACLED | 2010–present | District × event |
| IPC | 2010–present | District × season |
| WFP | 2010–present | Market × month |
| CHIRPS | 2010–present | District × month |
| NDVI | 2010–present | District × month |
| UNHCR | 2010–present | Country × month |
| FSNAU | 2010–present (sparse) | District × survey |
| Shapefiles | Static | District polygons |
