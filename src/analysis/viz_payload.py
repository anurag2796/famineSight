"""
Build a single chart-ready JSON payload (`models/viz_payload.json`)
that the Streamlit frontend reads to render every syllabus-mapped figure
without needing scikit-learn or any heavy deps at runtime.

Runs after the four core analyses (preprocessor, association, clustering,
classification, anomaly) have populated `data/processed/` and `models/`.

All numeric arrays are downsampled where reasonable (max ~5k points per
trace) so the JSON stays under a few MB.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors

from src.config import DATA_PROC, MODELS_DIR

logger = logging.getLogger(__name__)
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_list(arr) -> List:
    """Convert numpy/pandas to plain Python list (JSON-safe)."""
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _downsample(arr, n_max: int = 5000):
    arr = np.asarray(arr)
    if len(arr) <= n_max:
        return arr
    idx = RNG.choice(len(arr), size=n_max, replace=False)
    return arr[idx]


def _safe(fn, default=None, label="step"):
    try:
        return fn()
    except Exception as e:  # pragma: no cover - viz is best-effort
        logger.warning(f"viz step '{label}' failed: {e}")
        return default


# ---------------------------------------------------------------------------
# Per-chapter builders
# ---------------------------------------------------------------------------
def build_data_quality(master: pd.DataFrame) -> Dict[str, Any]:
    """Ch 2.2 — missingness + outlier rates per feature."""
    feature_cols = [
        c for c in master.columns
        if pd.api.types.is_numeric_dtype(master[c])
        and c not in ("crisis_label", "cdr_per_10k_per_day", "u5dr_per_10k_per_day")
    ]
    miss_pct = (master[feature_cols].isna().mean() * 100).round(2)

    # Coverage matrix: feature × year-month, fraction of districts with data
    if "date" in master.columns:
        month = pd.to_datetime(master["date"]).dt.to_period("M").astype(str)
        coverage = (
            master.assign(_month=month)
            .groupby("_month")[feature_cols]
            .apply(lambda g: g.notna().mean())
            .round(3)
        )
        cov_payload = {
            "months": coverage.index.tolist(),
            "features": feature_cols,
            "matrix": coverage.values.tolist(),
        }
    else:
        cov_payload = None

    # Outlier rate (Tukey 1.5*IQR) per feature
    outlier_rate = {}
    for c in feature_cols:
        s = master[c].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            outlier_rate[c] = 0.0
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_rate[c] = float(((s < lo) | (s > hi)).mean() * 100)

    return {
        "missing_pct": miss_pct.to_dict(),
        "outlier_pct": outlier_rate,
        "coverage": cov_payload,
        "n_rows": int(len(master)),
        "n_features": len(feature_cols),
        "n_districts": int(master["pcode"].nunique()) if "pcode" in master.columns else 0,
    }


def build_feature_engineering(master: pd.DataFrame, scaled: pd.DataFrame) -> Dict[str, Any]:
    """Ch 2.3 — discretization, transformation, lag-autocorrelation."""
    target = "food_price_index" if "food_price_index" in master.columns else None
    discretization = None
    if target:
        s = master[target].dropna()
        if len(s) > 50:
            ew = pd.cut(s, bins=5, labels=False, include_lowest=True).value_counts().sort_index()
            ef = pd.qcut(s, q=5, labels=False, duplicates="drop").value_counts().sort_index()
            discretization = {
                "feature": target,
                "raw_hist": _to_list(_downsample(s.values, 2000)),
                "equal_width": {"bins": list(range(len(ew))), "counts": ew.tolist()},
                "equal_freq": {"bins": list(range(len(ef))), "counts": ef.tolist()},
            }

    # Variable transformation: raw vs MinMax for the same column
    transform = None
    if target and target in scaled.columns:
        raw = master[target].dropna().values
        minmax = scaled[target].dropna().values
        log_t = np.log1p(np.clip(raw, a_min=0, a_max=None))
        transform = {
            "feature": target,
            "raw": _to_list(_downsample(raw, 2000)),
            "minmax": _to_list(_downsample(minmax, 2000)),
            "log": _to_list(_downsample(log_t, 2000)),
        }

    # Lag-autocorrelation heatmap (per district mean ACF for top features)
    acf_payload = None
    candidates = [
        c for c in ("food_price_index", "rainfall_anomaly_pct",
                    "conflict_fatalities", "ndvi_anomaly")
        if c in master.columns
    ]
    if candidates and "pcode" in master.columns and "date" in master.columns:
        max_lag = 6
        rows = []
        for c in candidates:
            acfs = []
            for _, g in master.sort_values("date").groupby("pcode"):
                s = g[c].astype(float).reset_index(drop=True)
                if s.std() == 0 or len(s) < max_lag + 2:
                    continue
                acfs.append([s.autocorr(lag=k) for k in range(1, max_lag + 1)])
            if acfs:
                rows.append(np.nanmean(np.array(acfs), axis=0).tolist())
            else:
                rows.append([0.0] * max_lag)
        acf_payload = {
            "features": candidates,
            "lags": list(range(1, max_lag + 1)),
            "matrix": rows,
        }

    return {
        "discretization": discretization,
        "transformation": transform,
        "lag_autocorr": acf_payload,
    }


def build_eda(master: pd.DataFrame, pca_df: pd.DataFrame) -> Dict[str, Any]:
    """Ch 3 — summary stats + parallel coords + t-SNE."""
    feat_cols = [
        c for c in master.columns
        if pd.api.types.is_numeric_dtype(master[c])
        and c not in ("crisis_label", "cdr_per_10k_per_day", "u5dr_per_10k_per_day")
    ]
    desc = master[feat_cols].describe().T.round(3)
    desc_payload = {
        "features": desc.index.tolist(),
        "columns": desc.columns.tolist(),
        "matrix": desc.values.tolist(),
    }

    # Parallel coordinates: top-6 most-variable features × 1500 sampled rows
    top6 = master[feat_cols].std().sort_values(ascending=False).head(6).index.tolist()
    sample = master[top6 + ["crisis_label"]].dropna()
    if len(sample) > 1500:
        sample = sample.sample(1500, random_state=42)
    parallel = {
        "features": top6,
        "rows": sample[top6].values.tolist(),
        "label": sample["crisis_label"].astype(int).tolist(),
    }

    # t-SNE on PCA components (cheap)
    tsne_payload = None
    pc_cols = [c for c in pca_df.columns if c.startswith("pca_comp_")]
    if pc_cols and len(pca_df) > 50:
        n = min(2000, len(pca_df))
        sub = pca_df.sample(n=n, random_state=42)
        try:
            X = sub[pc_cols].values
            tsne = TSNE(
                n_components=2, perplexity=min(30, n // 4),
                init="pca", random_state=42, max_iter=500,
            )
            coords = tsne.fit_transform(X)
            tsne_payload = {
                "x": coords[:, 0].tolist(),
                "y": coords[:, 1].tolist(),
                "label": sub.get("crisis_label", pd.Series([0] * n)).astype(int).tolist(),
                "pcode": sub.get("pcode", pd.Series([""] * n)).tolist(),
            }
        except Exception as e:
            logger.warning(f"t-SNE failed: {e}")

    # Crisis rate by year (OLAP roll-up)
    olap = None
    if "date" in master.columns:
        m = master.copy()
        m["year"] = pd.to_datetime(m["date"]).dt.year
        olap_table = (
            m.groupby(["year"])
            .agg(crisis_rate=("crisis_label", "mean"),
                 n_records=("crisis_label", "size"))
            .reset_index()
        )
        olap = olap_table.to_dict("records")

    return {
        "describe": desc_payload,
        "parallel": parallel,
        "tsne": tsne_payload,
        "olap_year": olap,
    }


def build_pca_diagnostics() -> Dict[str, Any]:
    """Appendix B — scree + cumulative variance from saved PCA model."""
    pca_path = MODELS_DIR / "pca.joblib"
    if not pca_path.exists():
        return {}
    pca = joblib.load(pca_path)
    evr = getattr(pca, "explained_variance_ratio_", None)
    if evr is None:
        return {}
    cum = np.cumsum(evr)
    return {
        "explained_variance": evr.tolist(),
        "cumulative": cum.tolist(),
        "n_components": int(len(evr)),
    }


def build_feature_selection(master: pd.DataFrame) -> Dict[str, Any]:
    """Ch 2.3.4 — chi-square + mutual information rankings."""
    feat_cols = [
        c for c in master.columns
        if pd.api.types.is_numeric_dtype(master[c])
        and c not in ("crisis_label", "cdr_per_10k_per_day", "u5dr_per_10k_per_day")
    ]
    df = master[feat_cols + ["crisis_label"]].dropna()
    if len(df) < 50:
        return {}
    X = df[feat_cols].values
    y = df["crisis_label"].astype(int).values

    # chi2 needs non-negative values — shift each column to min 0
    X_pos = X - np.minimum(X.min(axis=0), 0)
    try:
        chi_scores = chi2(X_pos, y)[0]
    except Exception as e:
        logger.warning(f"chi2 failed: {e}")
        chi_scores = np.zeros(len(feat_cols))
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
    except Exception as e:
        logger.warning(f"mutual_info failed: {e}")
        mi_scores = np.zeros(len(feat_cols))

    return {
        "features": feat_cols,
        "chi2": [float(s) for s in chi_scores],
        "mutual_info": [float(s) for s in mi_scores],
    }


def build_classification_curves(pca_df: pd.DataFrame) -> Dict[str, Any]:
    """Ch 4.5/4.6 — ROC, PR, calibration, learning curves for each saved model."""
    out = {}
    target = "crisis_label"
    if target not in pca_df.columns:
        return out

    # Recreate the same temporal split logic the training script used
    df_sorted = pca_df.sort_values(["date", "pcode"]).reset_index(drop=True)
    unique_dates = df_sorted["date"].sort_values().unique()
    n = len(unique_dates)
    if n < 5:
        return out
    val_cut = unique_dates[int(n * 0.85)]
    test_df = df_sorted[df_sorted["date"] >= val_cut]

    feat_cols = [
        c for c in pca_df.columns
        if pd.api.types.is_numeric_dtype(pca_df[c])
        and c not in ("date", "pcode", "district", target,
                      "cdr_per_10k_per_day", "u5dr_per_10k_per_day")
    ]
    feat_cols = [c for c in feat_cols if pca_df[c].notna().any()]
    X_test = test_df[feat_cols].fillna(0).values
    y_test = test_df[target].astype(int).values

    if len(np.unique(y_test)) < 2:
        return {"note": "test set has a single class — curves not meaningful"}

    models = {}
    for name, fn in [
        ("random_forest",          "random_forest.joblib"),
        ("xgboost",                "xgboost_model.joblib"),
        ("decision_tree",          "decision_tree.joblib"),
        ("decision_tree_optimized","decision_tree_optimized.joblib"),
    ]:
        path = MODELS_DIR / fn
        if not path.exists():
            continue
        try:
            mdl = joblib.load(path)

            # Optimized DT uses a smaller lead_only feature set — rebuild X from
            # its stored feature list rather than the full pca_df feat_cols.
            if name == "decision_tree_optimized":
                stored_cols = getattr(mdl, "_famine_feature_cols", None)
                threshold   = getattr(mdl, "_famine_threshold", 0.5)
                if stored_cols:
                    avail = [c for c in stored_cols if c in test_df.columns]
                    X_use = test_df[avail].fillna(0).values
                else:
                    expected = getattr(mdl, "n_features_in_", X_test.shape[1])
                    X_use = X_test[:, :expected]
                    threshold = 0.5
            else:
                expected = getattr(mdl, "n_features_in_", X_test.shape[1])
                X_use = X_test
                if X_use.shape[1] < expected:
                    X_use = np.hstack([X_use, np.zeros((X_use.shape[0], expected - X_use.shape[1]))])
                elif X_use.shape[1] > expected:
                    X_use = X_use[:, :expected]
                threshold = 0.5

            proba = mdl.predict_proba(X_use)[:, 1] if hasattr(mdl, "predict_proba") else None
            if proba is None:
                continue
            fpr, tpr, _ = roc_curve(y_test, proba)
            prec, rec, _ = precision_recall_curve(y_test, proba)
            cal_true, cal_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
            cm = confusion_matrix(y_test, (proba >= threshold).astype(int)).tolist()
            models[name] = {
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
                "calibration": {"prob_true": cal_true.tolist(), "prob_pred": cal_pred.tolist()},
                "confusion_matrix": cm,
                "n_test": int(len(y_test)),
                "threshold": float(threshold),
            }
        except Exception as e:
            logger.warning(f"curves for {name} failed: {e}")

    # Pull comparison table and optimized DT rules from classification metadata
    meta_path = MODELS_DIR / "classification_metadata.joblib"
    if meta_path.exists():
        try:
            meta = joblib.load(meta_path)
            out["model_comparison"]     = meta.get("model_comparison", [])
            out["dt_optimized_rules"]   = meta.get("dt_optimized_rules", "")
            out["dt_optimized_threshold"] = meta.get("dt_optimized_threshold", 0.5)
        except Exception as e:
            logger.warning(f"Could not load classification metadata: {e}")

    out["models"] = models
    out["positive_rate"] = float(np.mean(y_test))
    return out


def build_split_demos() -> Dict[str, Any]:
    """Ch 4.3.4 — Gini vs Entropy as a function of class proportion p."""
    p = np.linspace(1e-3, 1 - 1e-3, 99)
    gini = 1 - p**2 - (1 - p) ** 2
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    cls_err = 1 - np.maximum(p, 1 - p)
    return {
        "p": p.tolist(),
        "gini": gini.tolist(),
        "entropy": entropy.tolist(),
        "classification_error": cls_err.tolist(),
    }


def build_clustering_extras(master: pd.DataFrame) -> Dict[str, Any]:
    """Ch 8.2 — elbow SSE, silhouette per k; Ch 8.3 — agglomerative dendrogram."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import MinMaxScaler

    feature_cols = [
        c for c in (
            "rainfall_anomaly_pct", "conflict_events", "conflict_fatalities",
            "civilian_targeting_events", "food_price_index",
            "ipc_phase1_pct", "ipc_phase2_pct", "ipc_phase3_pct",
            "ipc_phase4_pct", "ipc_phase5_pct",
        ) if c in master.columns
    ]
    if not feature_cols or "pcode" not in master.columns:
        return {}

    profile = master.groupby("pcode")[feature_cols].mean().fillna(0)
    if len(profile) < 4:
        return {}
    X = MinMaxScaler().fit_transform(profile.values)
    pcodes = profile.index.tolist()

    elbow = []
    for k in range(2, min(9, len(profile))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
        elbow.append({"k": k, "sse": float(km.inertia_), "silhouette": float(sil) if sil is not None else None})

    # Agglomerative linkage matrix (Ward)
    try:
        Z = linkage(X, method="ward")
        dendro = {"linkage": Z.tolist(), "labels": pcodes}
    except Exception as e:
        logger.warning(f"linkage failed: {e}")
        dendro = None

    return {
        "elbow": elbow,
        "dendrogram": dendro,
        "n_districts": len(pcodes),
    }


def build_anomaly_extras(master: pd.DataFrame) -> Dict[str, Any]:
    """Ch 10.2/10.3/10.4 — z-score histogram, kth-NN distance, LOF/IF score distributions."""
    out: Dict[str, Any] = {}

    # Z-scores for key features
    for col in ("food_price_index", "cdr_per_10k_per_day", "conflict_fatalities"):
        if col not in master.columns:
            continue
        s = master[col].dropna()
        if len(s) < 30:
            continue
        z = (s - s.mean()) / (s.std() + 1e-9)
        out.setdefault("zscore", {})[col] = {
            "values": _to_list(_downsample(z.values, 3000)),
        }

    # kth-NN distance (k=5) on standardised numeric features (Ch 10.3)
    feat_cols = [
        c for c in master.columns
        if pd.api.types.is_numeric_dtype(master[c])
        and c not in ("crisis_label", "cdr_per_10k_per_day", "u5dr_per_10k_per_day")
    ]
    df = master[feat_cols].dropna()
    if len(df) > 50:
        X = (df.values - df.values.mean(axis=0)) / (df.values.std(axis=0) + 1e-9)
        try:
            nn = NearestNeighbors(n_neighbors=6).fit(X)
            d, _ = nn.kneighbors(X)
            kth = d[:, -1]
            out["kth_nn"] = {
                "k": 5,
                "distances": _to_list(_downsample(kth, 3000)),
            }
        except Exception as e:
            logger.warning(f"kth-NN failed: {e}")

    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_payload(extra_models: Dict[str, Any] = None) -> Dict[str, Any]:
    """Read everything, build payload, save to MODELS_DIR/viz_payload.json.
    
    Args:
        extra_models: Optional dict with Phase-B model results (classifiers, clustering).
                     Will be merged into payload as 'extra_models' key if provided.
    """
    master_p = DATA_PROC / "master_panel.parquet"
    scaled_p = DATA_PROC / "panel_scaled.parquet"
    pca_p = DATA_PROC / "panel_pca.parquet"

    if not master_p.exists():
        raise FileNotFoundError(f"missing {master_p} — run preprocessor first")

    master = pd.read_parquet(master_p)
    scaled = pd.read_parquet(scaled_p) if scaled_p.exists() else master
    pca_df = pd.read_parquet(pca_p) if pca_p.exists() else master

    payload: Dict[str, Any] = {
        "data_quality": _safe(lambda: build_data_quality(master), {}, "data_quality"),
        "feature_engineering": _safe(lambda: build_feature_engineering(master, scaled), {}, "feature_eng"),
        "eda": _safe(lambda: build_eda(master, pca_df), {}, "eda"),
        "pca_diagnostics": _safe(build_pca_diagnostics, {}, "pca"),
        "feature_selection": _safe(lambda: build_feature_selection(master), {}, "fselect"),
        "classification_curves": _safe(lambda: build_classification_curves(pca_df), {}, "curves"),
        "split_demos": _safe(build_split_demos, {}, "split_demos"),
        "clustering_extras": _safe(lambda: build_clustering_extras(master), {}, "clust_extras"),
        "anomaly_extras": _safe(lambda: build_anomaly_extras(master), {}, "anom_extras"),
        "generated_at": pd.Timestamp.now().isoformat(),
    }

    # Merge Phase-B results if provided
    if extra_models:
        payload["extra_models"] = extra_models
        logger.info("extra_models merged into payload")

    out_path = MODELS_DIR / "viz_payload.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"viz_payload written to {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    
    # Verify extra_models persistence for integrity check
    if extra_models and "extra_models" not in payload:
        logger.error("CRITICAL: extra_models was not merged into final payload!")
    
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build_payload()
    print("viz_payload.json built successfully")
