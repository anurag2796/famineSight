# src/data/shapefile_fetcher.py
"""
Fetch Somalia administrative boundary data from OCHA's Common Operational Dataset
(COD-AB) on HDX — fully open under CC BY-IGO, no login required.

Source: https://data.humdata.org/dataset/cod-ab-som
Dataset: Somalia administrative level 0-2 boundaries (COD-AB) v03
  - Admin 1: 18 Regions
  - Admin 2: 91 Districts

We download the GeoJSON (pure Python, no geopandas / GDAL required) and:
  1. Save the raw Admin-2 GeoJSON for frontend map rendering.
  2. Extract a district lookup CSV: pcode, name, region, centroid_lat, centroid_lon.
"""

import json
import math
import logging
import requests
import zipfile
import io
from pathlib import Path
from src.config import DATA_RAW, DISTRICT_PCODES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HDX direct-download URLs (resource IDs from the COD-AB dataset page,
# last updated 26 January 2026).
# ---------------------------------------------------------------------------
GEOJSON_ZIP_URL = (
    "https://data.humdata.org/dataset/ec140a63-5330-4376-a3df-c7ebf73cfc3c"
    "/resource/79f7f826-6028-4650-b8f7-2d5e53032955"
    "/download/som_admin_boundaries.geojson.zip"
)

# Inside the zip, the Admin-2 file follows the OCHA naming convention:
ADMIN2_FILENAME = "som_admbnda_adm2_ocha_20251030.geojson"
ADMIN1_FILENAME = "som_admbnda_adm1_ocha_20251030.geojson"

OUTPUT_DIR = DATA_RAW / "shapefiles"


# ---------------------------------------------------------------------------
# Geometry helpers (no geopandas / shapely needed)
# ---------------------------------------------------------------------------

def _polygon_centroid(coords: list) -> tuple[float, float]:
    """
    Compute the centroid of a simple polygon using the shoelace / signed-area
    formula.  `coords` should be a list of [lon, lat] pairs forming a ring.
    Returns (lat, lon).
    """
    n = len(coords)
    if n == 0:
        return (0.0, 0.0)

    area = 0.0
    cx = 0.0
    cy = 0.0

    for i in range(n):
        x0, y0 = coords[i][0], coords[i][1]
        x1, y1 = coords[(i + 1) % n][0], coords[(i + 1) % n][1]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    area = area / 2.0
    if abs(area) < 1e-12:
        # Fallback: simple mean of vertices
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    cx = cx / (6.0 * area)
    cy = cy / (6.0 * area)
    return (cy, cx)  # (lat, lon)


def _feature_centroid(geometry: dict) -> tuple[float, float]:
    """
    Return (lat, lon) centroid for any GeoJSON geometry type.
    For multi-part geometries, uses the part with the largest bounding-box area.
    """
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        return _polygon_centroid(coords[0])  # outer ring

    elif gtype == "MultiPolygon":
        # Pick the largest polygon by rough bbox area
        best_ring = None
        best_size = -1.0
        for poly in coords:
            ring = poly[0]
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            size = (max(lons) - min(lons)) * (max(lats) - min(lats))
            if size > best_size:
                best_size = size
                best_ring = ring
        if best_ring:
            return _polygon_centroid(best_ring)
        return (0.0, 0.0)

    elif gtype == "Point":
        return (coords[1], coords[0])

    else:
        return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

def fetch_shapefile_data() -> dict:
    """
    Download OCHA COD-AB Somalia boundaries and save:
      - data/raw/shapefiles/som_admin2.geojson   (for map rendering)
      - data/raw/shapefiles/som_admin1.geojson
      - data/raw/shapefiles/district_lookup.csv  (pcode, name, region, lat, lon)

    Returns a dict with keys 'admin1_features', 'admin2_features', 'lookup_rows'
    containing counts; returns empty dict on failure.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading OCHA COD-AB GeoJSON from HDX …")
    try:
        resp = requests.get(GEOJSON_ZIP_URL, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        logger.error(f"Failed to download boundary data: {exc}")
        return {}

    logger.info(f"Downloaded {len(resp.content) / 1024:.1f} KB — extracting …")

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile as exc:
        logger.error(f"Downloaded file is not a valid ZIP: {exc}")
        return {}

    # List contents so we can find files even if names change slightly
    names = zf.namelist()
    logger.info(f"ZIP contents: {names}")

    # Find admin1 and admin2 geojson files
    # Prefer exact standard names, fall back to any file containing 'admin2'/'adm2'
    def _find(names, *hints):
        for hint in hints:
            match = next((n for n in names if hint in n.lower() and n.endswith(".geojson")), None)
            if match:
                return match
        return None

    adm2_name = _find(names, "som_admin2.geojson", "adm2")
    adm1_name = _find(names, "som_admin1.geojson", "adm1")

    # Exclude edge-matched variants (_em) — prefer plain boundaries
    def _prefer_non_em(names, base_hint):
        plain = next((n for n in names if base_hint in n and "_em" not in n and n.endswith(".geojson")), None)
        em    = next((n for n in names if base_hint in n and n.endswith(".geojson")), None)
        return plain or em

    adm2_name = _prefer_non_em(names, "admin2")
    adm1_name = _prefer_non_em(names, "admin1")

    if not adm2_name:
        logger.error("Could not find Admin-2 GeoJSON inside the ZIP.")
        logger.error(f"Available files: {names}")
        return {}

    # ---- Admin 2 -------------------------------------------------------
    raw_adm2 = json.loads(zf.read(adm2_name))
    adm2_path = OUTPUT_DIR / "som_admin2.geojson"
    with open(adm2_path, "w", encoding="utf-8") as fh:
        json.dump(raw_adm2, fh, ensure_ascii=False)
    logger.info(f"Saved Admin-2 GeoJSON → {adm2_path}")

    # ---- Admin 1 -------------------------------------------------------
    adm1_features_count = 0
    if adm1_name:
        raw_adm1 = json.loads(zf.read(adm1_name))
        adm1_path = OUTPUT_DIR / "som_admin1.geojson"
        with open(adm1_path, "w", encoding="utf-8") as fh:
            json.dump(raw_adm1, fh, ensure_ascii=False)
        adm1_features_count = len(raw_adm1.get("features", []))
        logger.info(f"Saved Admin-1 GeoJSON → {adm1_path}")

    # ---- District lookup CSV -------------------------------------------
    lookup_rows = []
    for feat in raw_adm2.get("features", []):
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})

        # OCHA COD-AB v03 field names (lowercase snake_case in this version)
        # Fall back through older naming conventions for forward-compatibility
        pcode  = (
            props.get("adm2_pcode") or
            props.get("ADM2_PCODE") or
            props.get("admin2Pcode") or ""
        )
        name   = (
            props.get("adm2_name") or
            props.get("adm2_ref_name") or
            props.get("ADM2_EN") or
            props.get("ADM2_NAME") or ""
        )
        region = (
            props.get("adm1_name") or
            props.get("ADM1_EN") or
            props.get("ADM1_NAME") or ""
        )
        region_pcode = (
            props.get("adm1_pcode") or
            props.get("ADM1_PCODE") or
            props.get("admin1Pcode") or ""
        )

        # Use OCHA-precomputed centre coordinates when available
        lat = props.get("center_lat", 0.0) or 0.0
        lon = props.get("center_lon", 0.0) or 0.0
        if lat == 0.0 and lon == 0.0 and geom:
            try:
                lat, lon = _feature_centroid(geom)
            except Exception:
                pass

        lookup_rows.append({
            "pcode": pcode,
            "name": name,
            "region": region,
            "region_pcode": region_pcode,
            "centroid_lat": round(lat, 5),
            "centroid_lon": round(lon, 5),
        })

    # Save as CSV
    import csv
    lookup_path = OUTPUT_DIR / "district_lookup.csv"
    with open(lookup_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["pcode", "name", "region", "region_pcode",
                        "centroid_lat", "centroid_lon"],
        )
        writer.writeheader()
        writer.writerows(lookup_rows)

    logger.info(
        f"Saved district lookup → {lookup_path} ({len(lookup_rows)} districts)"
    )

    # ---- P-code crosswalk: OCHA standard ↔ internal project P-codes --------
    # The project uses a simplified 20-zone internal taxonomy (SO0001–SO0020)
    # while OCHA uses 91 Admin-2 districts. This explicit map assigns each OCHA
    # district to the closest internal zone (many OCHA districts → one internal).
    # This crosswalk is used by the map API to colour-code prediction results.
    OCHA_TO_INTERNAL: dict[str, str] = {
        # Mogadishu sub-districts → SO0001
        "Bondhere": "SO0001", "Cabdulasis": "SO0001", "Daynile": "SO0001",
        "Dharkenley": "SO0001", "Hamar Jabjab": "SO0001", "Hamar Weyne": "SO0001",
        "Hawl Wadaag": "SO0001", "Heliwa": "SO0001", "Hodan": "SO0001",
        "Kahda": "SO0001", "Karaan": "SO0001", "Shangaani": "SO0001",
        "Shibis": "SO0001", "Waaberi": "SO0001", "Wadajir (Medina)": "SO0001",
        "Wardhigley": "SO0001", "Yaaqshid": "SO0001",
        # Kismayo area → SO0002
        "Kismaayo": "SO0002", "Afmadow": "SO0002", "Badhaadhe": "SO0002",
        # Baidoa / Bay region → SO0003
        "Buur Hakaba": "SO0003", "Diinsoor": "SO0003", "Qansax Dheere": "SO0003",
        # Afgooye / Lower Shabelle → SO0004
        "Baraawe": "SO0004", "Kurtunwaarey": "SO0004", "Marka": "SO0004",
        "Qoryooley": "SO0004", "Sablaale": "SO0004", "Wanla Weyn": "SO0004",
        # Luuq / Gedo region → SO0005
        "Baardheere": "SO0005", "Belet Xaawo": "SO0005", "Ceel Waaq": "SO0005",
        "Doolow": "SO0005", "Garbahaarey": "SO0005",
        # Hargeisa → SO0006
        "Hargeysa": "SO0006", "Gebiley": "SO0006",
        # Berbera / NW coast → SO0007
        "Berbera": "SO0007",
        # Galkayo (Gaalkacyo) → SO0008
        "Gaalkacyo": "SO0008", "Galdogob": "SO0008",
        # Gedo region (supplementary) → SO0010
        "Buur Hakaba": "SO0010",
        # Jamaame / Lower Juba → SO0011
        "Bu'aale": "SO0011", "Jilib": "SO0011", "Saakow": "SO0011",
        # Buurhakaba → SO0012  (already covered above, keep for clarity)
        # Dhuusamarreeb / Galgaduud → SO0013
        "Cabudwaaq": "SO0013", "Cadaado": "SO0013",
        "Ceel Buur": "SO0013", "Ceel Dheer": "SO0013",
        # Mudug → SO0014
        "Hobyo": "SO0014", "Jariiban": "SO0014", "Xarardheere": "SO0014",
        # Sanaag → SO0015
        "Ceel Afweyn": "SO0015", "Ceerigaabo": "SO0015", "Laasqoray": "SO0015",
        # Togdheer → SO0016
        "Burco": "SO0016", "Buuhoodle": "SO0016",
        "Owdweyne": "SO0016", "Sheikh": "SO0016",
        # Sool → SO0017
        "Caynabo": "SO0017", "Laas Caanood": "SO0017",
        "Taleex": "SO0017", "Xudun": "SO0017",
        # Bay region → SO0018
        "Baidoa": "SO0018",
        # Galgaduud → SO0019
        "Adan Yabaal": "SO0019", "Balcad": "SO0019",
        "Cadale": "SO0019", "Jowhar": "SO0019",
        # Hiiraan → SO0020
        "Belet Weyne": "SO0020", "Bulo Burto": "SO0020",
        "Jalalaqsi": "SO0020",
        # Awdal / NW → SO0006 (Hargeisa zone)
        "Baki": "SO0006", "Borama": "SO0006", "Lughaye": "SO0006", "Zeylac": "SO0006",
        # Bakool → SO0005 (Gedo/Luuq zone)
        "Ceel Barde": "SO0005", "Rab Dhuure": "SO0005",
        "Tayeeglow": "SO0005", "Waajid": "SO0005", "Xudur": "SO0005",
        # Bossaso / Puntland coast → SO0008 (Galkayo zone)
        "Bandarbeyla": "SO0008", "Bossaso": "SO0008", "Caluula": "SO0008",
        "Iskushuban": "SO0008", "Qandala": "SO0008", "Qardho": "SO0008",
        # Nugaal → SO0008
        "Burtinle": "SO0008", "Eyl": "SO0008", "Garoowe": "SO0008",
        # Remaining variants / alternate spellings
        "Baydhaba": "SO0003",        # Baidoa alternate spelling
        "Dhuusamarreeb": "SO0013",   # exact match to internal name
        "Luuq": "SO0005",            # exact match
        "Jamaame": "SO0011",         # exact match
        "Afgooye": "SO0004",         # exact match
        "Unspecified": "SO0001",     # Mogadishu catch-all
    }

    crosswalk_rows = []
    unmatched = []
    for row in lookup_rows:
        internal_pcode = OCHA_TO_INTERNAL.get(row["name"], "")
        if not internal_pcode:
            unmatched.append(row["name"])
        crosswalk_rows.append({
            "ocha_pcode":     row["pcode"],
            "ocha_name":      row["name"],
            "ocha_region":    row["region"],
            "internal_pcode": internal_pcode,
            "centroid_lat":   row["centroid_lat"],
            "centroid_lon":   row["centroid_lon"],
        })

    matched = sum(1 for r in crosswalk_rows if r["internal_pcode"])
    crosswalk_path = OUTPUT_DIR / "pcode_crosswalk.csv"
    with open(crosswalk_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["ocha_pcode", "ocha_name", "ocha_region",
                        "internal_pcode", "centroid_lat", "centroid_lon"],
        )
        writer.writeheader()
        writer.writerows(crosswalk_rows)

    logger.info(
        f"Saved P-code crosswalk → {crosswalk_path} "
        f"({matched}/{len(crosswalk_rows)} OCHA districts matched to internal codes)"
    )
    if unmatched:
        logger.info(f"Unmatched OCHA districts: {unmatched}")

    return {
        "admin1_features": adm1_features_count,
        "admin2_features": len(raw_adm2.get("features", [])),
        "lookup_rows": len(lookup_rows),
    }


def main():
    result = fetch_shapefile_data()
    if result:
        logger.info(
            f"SUCCESS — Admin1: {result['admin1_features']} regions, "
            f"Admin2: {result['admin2_features']} districts, "
            f"Lookup rows: {result['lookup_rows']}"
        )
    else:
        logger.error("Shapefile fetch FAILED")


if __name__ == "__main__":
    main()
