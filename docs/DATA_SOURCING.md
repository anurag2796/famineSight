# FamineSight Data Sourcing Guide

This document provides exact instructions on how to obtain the raw data required for the FamineSight prediction models, either through manual CSV downloads or via API integrations.

## 1. IPC (Integrated Food Security Phase Classification)
**Target Metric:** Acute Food Insecurity (AFI) Phases 1-5

* **Manual Download Link:** [IPC Global Platform - Somalia](https://www.ipcinfo.org/ipc-country-analysis/)
  * *Instructions:* Navigate to the Somalia country page, ensure you are looking at the **Acute Food Insecurity (AFI)** analysis (not Malnutrition), and click the "Download Data" or "Export CSV" button. **Crucial: Ensure your download includes historical data from January 2010 to Present.**
  * *Save Path:* `data/raw/ipc/ipc_phases.csv`
* **API Access:** [IPC API Request Portal](https://www.ipcinfo.org/ipc-api/)
  * *Key Requirement:* Free. You must register and request an API token through their portal.

## 2. ACLED (Armed Conflict Location & Event Data)
**Target Metric:** Conflict events, fatalities, civilian targeting

* **Manual Download Link:** [ACLED Data Export Tool](https://acleddata.com/data-export-tool/)
  * *Instructions:* Filter for Region: Africa ➔ Country: Somalia. Select the exact date range: **January 1, 2010 to Present**. For Event Types, leave the filter blank (or select all) so the pipeline can calculate total conflict intensity. Click Export.
  * *Save Path:* `data/raw/acled/somalia_acled_raw.csv`
* **API Access:** [ACLED Developer Portal](https://developer.acleddata.com/)
  * *Key Requirement:* Free for humanitarian/academic use. Requires registering an email and generating an API Password.

## 3. WFP (World Food Programme)
**Target Metric:** Food Prices (Market prices for cereals, etc.)

* **Manual Download Link:** [HDX WFP Food Prices - Somalia](https://data.humdata.org/dataset/26727d1b-af49-4323-9215-c2ac479abb87)
  * *Instructions:* Click on the CSV resource named "wfp_food_prices_som.csv" to download. HDX usually bundles all historical data natively, but verify it goes back to **at least 2010**.
  * *Save Path:* `data/raw/wfp/wfp_prices_som_raw.csv`
* **API Access:** HDX (Humanitarian Data Exchange)
  * *Key Requirement:* **No API Key Required.** *(Note: FamineSight is already configured to fetch this automatically.)*

## 4. CHIRPS (Climate Hazards Group InfraRed Precipitation)
**Target Metric:** Monthly Rainfall (mm)

* **Manual Download Link:** [HDX Somalia Climate Data](https://data.humdata.org/dataset/somalia-climate) or [UC Santa Barbara Archive](https://data.chc.ucsb.edu/products/CHIRPS-2.0/)
  * *Instructions:* Download the CSV files corresponding to Somalia district-level precipitation. You will need to pull the archive spanning **2010 to Present**.
  * *Save Path:* `data/raw/chirps/chirps_rainfall.csv`
* **API Access:** UCSB / HDX
  * *Key Requirement:* **No API Key Required.** Free and open access.

## 5. FSNAU (Food Security and Nutrition Analysis Unit)
**Target Metric:** Crude Death Rate (CDR) / Mortality

* **Manual Download Link:** No direct public CSV link.
  * *Instructions:* Data must be requested directly from FAO/FSNAU via their data sharing agreements, or extracted manually from the tables in their PDF reports published on [fsnau.org](https://fsnau.org/). You must request longitudinal data covering **2010 to Present**.
  * *Save Path:* `data/raw/fsnau/fsnau_mortality.csv`
* **API Access:** None available.
  * *Key Requirement:* Institutional access required.

---

### How to Use Manually Downloaded Data

Once you download these files and save them to the exact paths listed above, the data is ready for the pipeline. Our scripts will process the raw files and structure them for the machine learning models.
