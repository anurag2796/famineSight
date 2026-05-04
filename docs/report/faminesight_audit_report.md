# FamineSight — Comprehensive Audit Report
**Date:** 2026-05-01  
**Auditor:** Antigravity  
**Hardware Target:** Jetson AGX Orin 64GB (ARM64/aarch64)  
**Repository:** `/home/lab/codebase/projects/personalProjects/famineSight`

---

## Executive Summary

FamineSight is a humanitarian data mining system designed to predict hunger-related mortality in Somalia. The codebase has a **solid architectural foundation** — all 5 analysis modules are written, the FastAPI backend scaffolding is in place, and synthetic data flows through the full ML pipeline. However, **the system is NOT production-ready** today. There are multiple critical bugs that will cause the backend to crash on startup, a pervasive feature-dimension mismatch that makes every prediction endpoint broken, and several key deliverables (Dockerfiles, frontend, train script, test suite, notebooks) that are missing entirely.

**Overall Production Readiness: 35% / 100**

---

## 1. Critical Bugs (Will Break at Runtime)

### 🔴 BUG-01: Model Path Mismatch — Backend Cannot Start

**Severity:** CRITICAL | **File:** `backend/services/model_registry.py`

The `ModelRegistry` loads all models from `MODELS_DIR = ROOT / "models"`, which maps to `famineSight/models/`. However, **every training script saves models to `data/processed/*.joblib`** — a completely different directory. The `models/` directory is empty.

**Result:** The FastAPI backend throws an exception during the lifespan startup hook and refuses to serve any request.

```
# model_registry.py loads from:
MODELS_DIR / "random_forest.joblib"  → models/random_forest.joblib  (MISSING)

# Training scripts save to:
'data/processed/random_forest.joblib'   (EXISTS)
'data/processed/xgboost_model.joblib'   (EXISTS)
```

**Fix:** Either update `MODELS_DIR` in `config.py` to point to `data/processed/`, or update all training scripts to save to `models/`.

---

### 🔴 BUG-02: XGBoost Model Missing — Startup Failure

**Severity:** CRITICAL | **File:** `data/processed/`

The XGBoost model file `xgboost_model.joblib` does **not exist** on disk. The training code in `classification.py` uses `early_stopping_rounds` without specifying `verbose=0` explicitly and saves to `data/processed/xgboost_model.joblib`, but for some reason the file was never written.

```
❌ data/processed/xgboost_model.joblib  → MISSING
✅ data/processed/random_forest.joblib  → EXISTS (1.0 MB)
```

**Fix:** Re-run `scripts/train_pipeline.py --synthetic` (once BUG-01 is fixed).

---

### 🔴 BUG-03: Feature Dimension Mismatch in `/predict/mortality`

**Severity:** CRITICAL | **File:** `backend/routers/predict.py`

The prediction endpoint extracts **7 features** from the request and feeds them to a `MinMaxScaler` that was trained on **33 features**. Then it feeds the scaler output to a PCA that produces **4 components**, but the RandomForest was trained on **35 features**.

```
# predict.py builds:
feature_vector = [rainfall, conflict_fatalities, food_price, temp_anomaly, 
                  et_anomaly, ipc4_pct, ipc5_pct]  → shape (1, 7)

# But scaler expects: shape (1, 33)
# And RF expects:     shape (1, 35)
```

Every call to `POST /predict/mortality` will raise `ValueError: X has 7 features, but MinMaxScaler is expecting 33 features.`

**Fix:** The prediction endpoint must reconstruct the full 33-feature vector (all lag features, rolling means, and PCA components) from the raw inputs before calling the scaler, or expose a simpler model trained directly on raw inputs.

---

### 🔴 BUG-04: Missing Features in Trained Data vs. API Schema

**Severity:** CRITICAL | **File:** `backend/schemas/input.py`, `src/analysis/classification.py`

The API schema accepts `temperature_anomaly` and `evapotranspiration_anomaly`, but these fields **do not exist in the trained panel**. The synthetic data generator never produced them; the preprocessor never ingested them. The ML models have no knowledge of these features.

| Feature in API | In Panel | Config `CLIMATE_FEATURES` |
|---|---|---|
| `rainfall_anomaly_pct` | ✅ | ✅ |
| `temperature_anomaly` | ❌ | ✅ |
| `evapotranspiration_anomaly` | ❌ | ✅ |
| `ipc_phase4_pct` | ✅ | ✅ |
| `ipc_phase5_pct` | ✅ | ✅ |

Similarly, `inflation_rate` and `exchange_rate` are in `MARKET_FEATURES` config but absent from the panel.

**Fix:** Either add these features to the synthetic data generator and re-train, or remove them from the API schema.

---

### 🔴 BUG-05: False Negatives Computed from Wrong Confusion Matrix Index

**Severity:** CRITICAL (for humanitarian accuracy) | **File:** `src/analysis/classification.py`, line 133

The false negative count (missed crises = lives lost) is computed as `cm[0, 1]` which is actually the **False Positive** count. False Negatives (predicted "no crisis" when there actually was one) are at `cm[1, 0]`.

```python
# WRONG — this is False Positives (FP)
false_negatives = cm[0, 1]

# CORRECT — this is False Negatives (FN)
false_negatives = cm[1, 0]
```

In a humanitarian context, this metric is explicitly described as "lives lost." Reporting FPs as FNs is a serious factual error.

---

### 🟠 BUG-06: Temporal Split Is Not Temporal — Data Leakage Risk

**Severity:** HIGH | **File:** `src/analysis/classification.py`, function `temporal_split()`

Despite being named `temporal_split`, the function splits by **district identity** (first 80% of district pcodes go to train, etc.), NOT by time. Train, validation, and test sets will overlap in time period. Since time-series data has autocorrelation, this causes data leakage and inflates reported model performance metrics.

```python
# WRONG: splits by district index, not time
train_districts = districts[:n_train]   # SO0001–SO0016
val_districts   = districts[n_train:n_train+n_val]  # SO0017–SO0018
test_districts  = districts[n_train+n_val:]  # SO0019–SO0020
# All three sets have data from 2010–2024 simultaneously → LEAKAGE
```

**Correct approach:** Split by date (e.g., train ≤ 2021, val = 2022, test ≥ 2023) for all districts.

---

### 🟠 BUG-07: Data Quality Issues in Master Panel

**Severity:** HIGH | **File:** `data/processed/master_panel.parquet`

The pipeline reports:
- **912 missing values remain** after imputation — the preprocessor's imputation is not fully working
- **Crisis rate is 18.1%** — exceeds the specified 3–20% ceiling barely, but the CLAUDE.md target is 5–12%  
- **Rainfall-CDR correlation is -0.014** (very weak) — the synthetic data generator doesn't enforce a strong negative physics relationship between drought and mortality as required

---

### 🟠 BUG-08: `OLLAMA_MODEL` Duplicated in `.env` — Wrong Model Loaded

**Severity:** HIGH | **File:** `.env`

The `.env` file defines `OLLAMA_MODEL` twice:

```bash
OLLAMA_MODEL=mistral:7b   # line 4 — first occurrence
OLLAMA_MODEL=qwen3:32b    # line 9 — second occurrence
```

`python-dotenv` reads the **last** value, so `qwen3:32b` is active. But the duplicate is a maintenance hazard that could silently break when lines are reordered. Also, the `ACLED_API_KEY` variable referenced throughout CLAUDE.md does not exist in `.env`; it uses `ACLED_PASSWORD` instead, creating a disconnect between docs and implementation.

---

### 🟡 BUG-09: `narrative.py` Uses Pydantic v1 `.dict()` Method

**Severity:** MEDIUM | **File:** `backend/routers/narrative.py`, line 63

The code calls `request.prediction.dict()` which is the Pydantic **v1** API. The project uses Pydantic **v2.12.5**. While Pydantic v2 maintains backward-compat for `.dict()` with a deprecation warning, it will be removed and should be `request.prediction.model_dump()`.

---

### 🟡 BUG-10: LOF Model Is Transductive — Cannot Predict on New Data

**Severity:** MEDIUM | **File:** `src/analysis/anomaly.py`, `backend/services/model_registry.py`

`LocalOutlierFactor` in scikit-learn is **transductive** by default (no `.predict()` method). The registry loads and stores `lof_model.joblib` but there's no production inference path that uses it. For deployment, LOF must be initialized with `novelty=True` to support inference on new samples.

---

### 🟡 BUG-11: Health Endpoint Timestamp Type Mismatch

**Severity:** MEDIUM | **File:** `backend/main.py`, line 83

`timestamp=Path(__file__).stat().st_mtime` returns a `float` (Unix epoch). `HealthResponse.timestamp` is `datetime`. Pydantic v2 coerces this successfully, but the timestamp will be the file modification time of `main.py`, not the current time — which is semantically wrong for a health check.

---

### 🟡 BUG-12: `APRIORI_MIN_SUPPORT` Referenced but Not Defined in Config

**Severity:** MEDIUM | **File:** `src/config.py`, `src/analysis/association.py`

`association.py` references `APRIORI_MIN_SUPPORT` from config but the config only defines `FP_MIN_SUPPORT`. The two algorithms currently use the same support threshold — this is ambiguous. Apriori typically uses a higher minimum support than FP-Growth.

---

### 🟡 BUG-13: `run_sequential()` is a Stub — Returns Empty DataFrame

**Severity:** MEDIUM | **File:** `src/analysis/association.py`, lines 204–208

The sequential pattern mining function is a complete stub:

```python
def run_sequential(sequences):
    logger.info("Sequential pattern mining skipped (PrefixSpan not available)")
    return pd.DataFrame()
```

The CLAUDE.md specification requires PrefixSpan for discovering temporal crisis pathways (D→C→P→M). This is unimplemented and will show 0 sequential patterns in all API responses.

---

### 🟡 BUG-14: Association Results Cannot Be JSON-Serialized (DataFrames)

**Severity:** MEDIUM | **File:** `src/analysis/association.py`, `backend/services/model_registry.py`

`run_all()` returns `fp_rules` and `apriori_rules` as **Pandas DataFrames** and `sequential_patterns` as a DataFrame. These are stored as Python objects, never serialized to JSON. The model registry tries to load `association_results.json` which doesn't exist because no JSON serialization step was implemented.

---

### 🟡 BUG-15: `cluster_results.json` and `anomaly_results.json` Never Written

**Severity:** MEDIUM | **File:** `src/analysis/clustering.py`, `src/analysis/anomaly.py`

Neither clustering nor anomaly analysis saves JSON result files. The registry loads `cluster_results.json` and `anomaly_results.json` from disk — both missing — so `GET /analyze/clusters` and `GET /anomaly/alerts` always return empty results even after training completes.

---

### 🟡 BUG-16: Bare `except:` Clause in Health Check Silences All Errors

**Severity:** MEDIUM | **File:** `backend/main.py`, lines 72–77

```python
try:
    async with httpx.AsyncClient() as client:
        response = await client.get("http://host.docker.internal:11434/api/tags")
        ollama_available = response.status_code == 200
except:   # ← bare except catches EVERYTHING including KeyboardInterrupt
    ollama_available = False
```

The `ollama_available` variable is also set but **never returned** in the `HealthResponse`. The health endpoint always returns `model_available=registry._loaded` without reflecting Ollama status.

---

### 🟡 BUG-17: `panel_scaled.parquet` and `panel_pca.parquet` Are Identical to `master_panel.parquet`

**Severity:** MEDIUM | **Data pipeline integrity**

All three parquet files are bitwise identical (`df_raw.equals(df_scaled) == True`). The preprocessor's `scale_features()` and `apply_pca()` functions save their output but the main pipeline doesn't call them in the right order — the PCA components appear as columns in all three files, suggesting the PCA was merged into master rather than being applied as a transformation step.

---

## 2. Missing Deliverables (Not Yet Built)

| Deliverable | Status | Impact |
|---|---|---|
| `backend/Dockerfile` | ❌ Missing | Cannot containerize |
| `frontend/Dockerfile` | ❌ Missing | Cannot containerize |
| `docker-compose.yml` | ❌ Missing | Cannot deploy with Docker |
| `frontend/app.py` (Streamlit) | ❌ Missing | No dashboard |
| `backend/requirements.txt` | ❌ Missing | Cannot install deps |
| `frontend/requirements.txt` | ❌ Missing | Cannot install deps |
| `scripts/train_pipeline.py` | ❌ Missing | No one-shot training |
| `.dockerignore` | ❌ Missing | Docker context bloat |
| `backend/tests/test_predict.py` | ❌ Missing (only `__init__.py`) | No test coverage |
| `backend/tests/conftest.py` | ❌ Missing | No test fixtures |
| `backend/services/inference.py` | ❌ Missing | Spec says it should exist |
| `notebooks/*.ipynb` | ❌ Empty dir | No EDA notebooks |

---

## 3. Security Vulnerabilities

### 🔴 SEC-01: Credentials Hardcoded in `.env` (Committed to Disk)

**Severity:** CRITICAL

The `.env` file contains real, live credentials:
- **ACLED password:** `Coloreal@2026`
- **Groq API key:** `vgsk_Jur3iCwTZV3PRYp1eBfwWGdyb3FYzV3j5x1zLyLvwQjc8RQallD1`
- **ACLED email:** `al5150@g.rit.edu`

If `.gitignore` doesn't exclude `.env`, these credentials will be committed to version history and potentially exposed publicly.

**Fix:** Immediately:
1. Rotate the Groq API key
2. Change the ACLED password
3. Add `.env` to `.gitignore` and verify it's not tracked

---

### 🟠 SEC-02: CORS Wildcard (`allow_origins=["*"]`)

**Severity:** HIGH | **File:** `backend/main.py`

The FastAPI backend allows requests from any origin with credentials. In production, this should be restricted to the Streamlit frontend's domain/IP only.

---

### 🟠 SEC-03: No Authentication on Any API Endpoint

**Severity:** HIGH | **File:** All routers

All API endpoints (`/predict/mortality`, `/narrative/generate`, `/anomaly/alerts`, etc.) are completely unauthenticated. The `/narrative/generate` endpoint will execute the local Ollama model for any caller. The system is intended as an internal tool, so at minimum HTTP Basic Auth or a static API key should be added.

---

## 4. Architecture & Design Issues

### 🟠 ARCH-01: No Proper Separation Between Training-Time and Inference-Time Models

The training pipeline (`classification.py`) applies an internal `StandardScaler` and uses those scaled arrays to train RF/XGB, but the `model_registry.py` loads a separately-saved `MinMaxScaler` (`scaler.joblib`) from the preprocessor pipeline. These are **different scalers** for different purposes. The predict endpoint's pipeline (scaler → PCA → RF) does not match the training pipeline (KNNImputer → internal StandardScaler → RF without PCA).

### 🟠 ARCH-02: Hardcoded Paths Instead of Using Config Constants

Training code saves models with hardcoded strings like `'data/processed/random_forest.joblib'` instead of `str(MODELS_DIR / 'random_forest.joblib')`. This causes the path mismatch described in BUG-01.

### 🟠 ARCH-03: `backend/services/inference.py` — Stub Missing

The CLAUDE.md specification includes `inference.py` as a required service file, but only `model_registry.py` exists. The inference logic that should live there is duplicated across all 4 routers.

### 🟡 ARCH-04: `OllamaClient` is Initialized at Module Import Time

`hybrid_client = HybridClient()` and `ollama_client = hybrid_client.ollama_client` are instantiated when the module is imported. The `httpx.AsyncClient` embedded in `OllamaClient.__init__` is created synchronously and shared across requests without proper lifecycle management. This will cause resource leaks in production.

### 🟡 ARCH-05: `ModelRegistry` Singleton Pattern Is Thread-Unsafe

The singleton uses `_instance = None` at the class level, but Python's `__new__` without a lock is not thread-safe in a multi-worker uvicorn deployment. Should use `threading.Lock()` or FastAPI's `State` for dependency injection.

---

## 5. Code Quality Issues

| Issue | File | Severity |
|---|---|---|
| Unused import: `Depends`, `HTTPException` in `predict.py` | `backend/routers/predict.py:2` | Low |
| Unused import: `HealthResponse`, `ClusterProfile`, `MortalityPrediction` in `main.py` | `backend/main.py:14` | Low |
| Unused import: `StandardScaler`, `KNNImputer`, `Pipeline`, `resample` in `classification.py` | `src/analysis/classification.py` | Low |
| Unused variable `district_means`, `district_stds` in z-score loop | `src/analysis/anomaly.py:178-179` | Low |
| Unused variable `numeric_cols` declared but shadow-overwritten in anomaly.py | `src/analysis/anomaly.py:31, 97, 159` | Low |
| `run_full_pipeline` imported in `audit.py` but never called (phases just re-read files) | `scripts/audit.py:18` | Low |
| `load_and_merge` imported in `association.py` but never called | `src/analysis/association.py:11` | Low |
| Cluster label names (0=Chronically Vulnerable, etc.) are arbitrary and not data-driven | `src/analysis/clustering.py:118-123` | Medium |
| `SHAP` called on RF model trained on scaled data but sample is from scaled test set — OK, but no error handling for multi-output SHAP | `src/analysis/classification.py:274` | Medium |

---

## 6. Production Readiness Assessment

| Component | Status | Notes |
|---|---|---|
| **Configuration** | 🟡 Partial | RF_N_JOBS=4, XGB cpu — ✅. Missing APRIORI_MIN_SUPPORT, ACLED_API_KEY |
| **Data Pipeline** | 🟡 Partial | Synthetic data works. 912 missing values remain. panel_scaled ≠ panel_master |
| **Association Rules** | 🟡 Partial | FP-Growth & Apriori run. Sequential mining is a stub. No JSON serialization |
| **Clustering** | 🟢 Working | KMeans + DBSCAN run, models saved. JSON not written |
| **Classification** | 🔴 Broken | False negative index wrong. Temporal split is district-split. XGB model missing |
| **Anomaly Detection** | 🟡 Partial | IsoForest + Z-score work. LOF is transductive. JSON not written |
| **LLM Client** | 🟢 Working | Ollama + Groq hybrid client well-designed |
| **Guardrails** | 🟡 Partial | Blocks 'crisis' in LOW risk (overly aggressive). Missing probability mismatch test |
| **FastAPI Backend** | 🔴 Broken | Won't start (empty models/). Predict endpoint crashes on every call |
| **Streamlit Frontend** | 🔴 Missing | `frontend/` directory is empty |
| **Docker** | 🔴 Missing | No Dockerfiles, no docker-compose.yml |
| **Tests** | 🔴 Missing | No actual test files exist |
| **Notebooks** | 🔴 Missing | `notebooks/` is empty |
| **Train Script** | 🔴 Missing | `scripts/train_pipeline.py` doesn't exist |
| **Security** | 🔴 Critical | Live credentials in `.env` |

---

## 7. Hardware Assessment — Jetson AGX Orin 64GB

The hardware is **well-suited** for this project with some caveats:

| Constraint | Status | Notes |
|---|---|---|
| `RF_N_JOBS=4` (not -1) | ✅ Enforced | Prevents OOM on 12-core ARM64 |
| `XGB_DEVICE="cpu"` | ✅ Enforced | Prevents CUDA kernel errors |
| `tree_method="hist"` in XGBoost | ✅ Present | CPU-optimized for ARM64 |
| GDAL apt dependencies | ⚠️ Spec only | Dockerfile is missing so untested |
| `extra_hosts: host-gateway` | ✅ In dev compose | Needed for Ollama connectivity |
| Memory ceiling for backend (4-8 GB) | ✅ In dev compose | Appropriate for this workload |

**Performance Projections on Jetson AGX Orin:**

| Task | Estimated Time | RAM Usage |
|---|---|---|
| Synthetic data generation | ~5 sec | < 500 MB |
| Preprocessing pipeline | ~15 sec | ~1 GB |
| RF training (100 estimators, 4 jobs) | ~2-4 min | ~3 GB |
| XGBoost training | ~30 sec | ~1 GB |
| IsoForest fitting | ~10 sec | ~500 MB |
| Qwen3:32b inference (first token) | ~8-15 sec | ~20-24 GB VRAM (unified) |
| Streamlit dashboard | ~2 sec to load | ~500 MB |

The 64GB unified memory is the primary asset. Running Qwen3:32b (≈ 20GB quantized) alongside a 4GB backend container, 1GB frontend, and OS overhead still leaves ~35GB free — comfortable.

---

## 8. Model Recommendations for This Hardware

### LLM Options (Ranked by Best Fit for Jetson AGX Orin 64GB)

| Model | Size | Quality | Latency | Recommendation |
|---|---|---|---|---|
| **Qwen3:32b (Q4_K_M)** | ~20 GB | ⭐⭐⭐⭐⭐ | ~8-15s/first token | 🥇 **Best overall** — fits comfortably, best reasoning at this scale for humanitarian analysis |
| **Llama3.1:70b (Q2_K)** | ~35 GB | ⭐⭐⭐⭐ | ~20-30s/first token | Marginal on 64GB; leaves only 29GB for rest of system |
| **Mistral:7b (Q4_K_M)** | ~4.5 GB | ⭐⭐⭐ | ~1-2s/first token | Fast but lower quality reasoning. Good as fallback |
| **Llama3.2:11b (Q4_K_M)** | ~7 GB | ⭐⭐⭐⭐ | ~2-4s/first token | Good balance of speed and quality for simpler narratives |
| **Groq API (llama3-8b)** | Cloud | ⭐⭐⭐ | ~0.5s | Zero GPU usage, but requires internet + API key cost |

**Current config uses Qwen3:32b — this is the right choice.** The `mistral:7b` in line 4 of `.env` is a leftover.

### ML Model Recommendations

| Current Model | Issue | Recommended Alternative |
|---|---|---|
| `RandomForestClassifier(n_jobs=4)` | Adequate | Consider `LightGBM` — 3-5× faster, less memory on ARM64 |
| `XGBClassifier(tree_method='hist')` | Good choice | Keep as-is |
| `IsolationForest` | Good for production inference | Keep |
| `LocalOutlierFactor` | Cannot do production inference | Replace with `novelty=True` or switch to `OneClassSVM` |
| `KMeans(k=4)` | Elbow not actually computed dynamically | Fine for current dataset size |

---

## 9. Prioritized Fix Roadmap

### Phase 1 — Critical (Must Fix Before Any Testing)

1. **Fix model save paths** — change all training scripts to save to `MODELS_DIR` (or change `MODELS_DIR` to `data/processed/`)
2. **Re-run training** — generate `xgboost_model.joblib` and `classification_metadata.joblib`
3. **Fix prediction endpoint** — align feature engineering between training and inference
4. **Add temperature/ET features to synthetic data** OR remove from API schema
5. **Write JSON serialization** for association, cluster, and anomaly results
6. **Rotate Groq API key and ACLED password** — treat as compromised

### Phase 2 — High Priority (Before Integration Testing)

7. **Fix false_negatives index** (`cm[1,0]` not `cm[0,1]`)
8. **Implement true temporal split** (date-based, not district-based)
9. **Write Dockerfiles** and `docker-compose.yml`
10. **Write `frontend/app.py`** (Streamlit dashboard)
11. **Write `backend/tests/`** (conftest, test_predict, test_anomaly)
12. **Write `scripts/train_pipeline.py`**

### Phase 3 — Medium Priority (Before Production)

13. **Fix missing values** (912 remaining after imputation)
14. **Implement PrefixSpan** sequential mining
15. **Fix LOF** for novelty detection (`novelty=True`)
16. **Fix CORS** to restrict to frontend origin
17. **Add API authentication**
18. **Fix `OLLAMA_MODEL` duplicate in `.env`**
19. **Implement `inference.py`** service
20. **Write 5 Jupyter notebooks**

---

## 10. Summary Scorecard

| Category | Score | Grade |
|---|---|---|
| Architecture Design | 65/100 | C+ |
| Code Correctness (bug-free) | 40/100 | D |
| Feature Completeness | 45/100 | D+ |
| Security | 20/100 | F |
| Test Coverage | 5/100 | F |
| Documentation | 70/100 | B- |
| Hardware Optimization | 85/100 | B+ |
| Data Pipeline | 55/100 | D+ |
| **Overall Production Readiness** | **35/100** | **F** |

> [!CAUTION]
> The system CANNOT be deployed to production in its current state. The backend will crash on startup due to missing model files, every prediction call will raise a dimension error, and live API credentials are exposed in the repository.

> [!IMPORTANT]
> The Jetson AGX Orin 64GB is an excellent hardware choice. Qwen3:32b is the right LLM. The ML architecture (RF + XGB + IsoForest + clustering) is sound. The core bugs are fixable in 2-3 days of focused work.

> [!TIP]
> The fastest path to a working demo is: (1) fix model paths, (2) align predict endpoint feature pipeline, (3) write JSON serialization for results, (4) write the Streamlit frontend. The rest of the system will work once these 4 items are done.
