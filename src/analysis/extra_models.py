"""
Syllabus-coverage Phase-B models (Tan/Steinbach/Kumar Ch 5, 8, 9, 10).

Trains lightweight reference implementations for chapters that the core
pipeline doesn't already cover (kNN, Naive Bayes, SVM, MLP, GMM, MST,
clustering-based anomaly), and appends their results to the existing
`models/viz_payload.json` produced by `viz_payload.py`.

Designed to be cheap on CPU — all models are capped to small grids /
sample sizes so the full module finishes in well under a minute on
synthetic data.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.config import DATA_PROC, MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train/test split — same logic as classification.run_all
# ---------------------------------------------------------------------------
def _split(pca_df: pd.DataFrame):
    df = pca_df.sort_values(["date", "pcode"]).reset_index(drop=True)
    unique_dates = df["date"].sort_values().unique()
    n = len(unique_dates)
    if n < 5:
        raise ValueError("not enough unique dates for split")
    train_cut = unique_dates[int(n * 0.70)]
    val_cut = unique_dates[int(n * 0.85)]
    train = df[df["date"] < train_cut]
    test = df[df["date"] >= val_cut]

    target = "crisis_label"
    feat_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in ("date", "pcode", "district", target,
                      "cdr_per_10k_per_day", "u5dr_per_10k_per_day")
    ]
    feat_cols = [c for c in feat_cols if df[c].notna().any()]

    X_train = train[feat_cols].fillna(0).values
    y_train = train[target].astype(int).values
    X_test = test[feat_cols].fillna(0).values
    y_test = test[target].astype(int).values
    return X_train, y_train, X_test, y_test, feat_cols


# ---------------------------------------------------------------------------
# Phase B classifiers (Ch 5)
# ---------------------------------------------------------------------------
def _evaluate(name: str, model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X_test)
        proba = (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        proba = y_pred.astype(float)

    out = {
        "name": name,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if len(np.unique(y_test)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_test, proba))
            out["avg_precision"] = float(average_precision_score(y_test, proba))
            fpr, tpr, _ = roc_curve(y_test, proba)
            prec, rec, _ = precision_recall_curve(y_test, proba)
            out["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            out["pr"] = {"precision": prec.tolist(), "recall": rec.tolist()}
        except Exception as e:
            logger.warning(f"{name} curves failed: {e}")
    return out


def train_extra_classifiers(pca_df: pd.DataFrame) -> Dict[str, Any]:
    """Ch 5.2 / 5.3 / 5.4 / 5.5 — kNN, Naive Bayes, MLP, SVM."""
    X_train, y_train, X_test, y_test, _ = _split(pca_df)

    # Subsample SVM training set if it's big — quadratic in n
    if len(X_train) > 4000:
        idx = np.random.RandomState(42).choice(len(X_train), 4000, replace=False)
        X_svm, y_svm = X_train[idx], y_train[idx]
    else:
        X_svm, y_svm = X_train, y_train

    results: Dict[str, Any] = {}

    # kNN — also save k-vs-accuracy curve (Ch 5.2)
    k_curve = []
    for k in [1, 3, 5, 7, 11, 15, 21, 31]:
        knn_k = KNeighborsClassifier(n_neighbors=k, n_jobs=1)
        knn_k.fit(X_train, y_train)
        acc = float((knn_k.predict(X_test) == y_test).mean())
        k_curve.append({"k": k, "accuracy": acc})
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1).fit(X_train, y_train)
    results["knn"] = _evaluate("k-Nearest Neighbors", knn, X_test, y_test)
    results["knn"]["k_curve"] = k_curve

    # Naive Bayes (Ch 5.3.3)
    nb = GaussianNB().fit(X_train, y_train)
    results["naive_bayes"] = _evaluate("Gaussian Naive Bayes", nb, X_test, y_test)

    # MLP (Ch 5.4) — small net, capped epochs
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=200,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(X_train, y_train)
    results["mlp"] = _evaluate("MLP (32,16)", mlp, X_test, y_test)
    if hasattr(mlp, "loss_curve_"):
        results["mlp"]["loss_curve"] = mlp.loss_curve_

    # SVM (Ch 5.5) — RBF kernel, prob=True for ROC
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE)
    try:
        svm.fit(X_svm, y_svm)
        results["svm"] = _evaluate("SVM (RBF)", svm, X_test, y_test)
        results["svm"]["n_support"] = [int(x) for x in svm.n_support_]
    except Exception as e:
        logger.warning(f"SVM training failed: {e}")
        results["svm"] = {"name": "SVM (RBF)", "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Phase B clustering (Ch 9)
# ---------------------------------------------------------------------------
def train_extra_clustering(master: pd.DataFrame) -> Dict[str, Any]:
    feat_cols = [
        c for c in (
            "rainfall_anomaly_pct", "conflict_events", "conflict_fatalities",
            "civilian_targeting_events", "food_price_index",
            "ipc_phase1_pct", "ipc_phase2_pct", "ipc_phase3_pct",
            "ipc_phase4_pct", "ipc_phase5_pct",
        ) if c in master.columns
    ]
    if not feat_cols or "pcode" not in master.columns:
        return {}

    profile = master.groupby("pcode")[feat_cols].mean().fillna(0)
    if len(profile) < 6:
        return {}
    pcodes = profile.index.tolist()
    X = MinMaxScaler().fit_transform(profile.values)

    # GMM (Ch 9.2) — log-likelihood for k=2..6 + final assignment for k=4
    gmm_k = []
    for k in range(2, 7):
        try:
            g = GaussianMixture(n_components=k, random_state=42).fit(X)
            gmm_k.append({
                "k": k,
                "bic": float(g.bic(X)),
                "aic": float(g.aic(X)),
            })
        except Exception as e:
            logger.warning(f"GMM k={k} failed: {e}")
    g = GaussianMixture(n_components=4, random_state=42).fit(X)
    gmm_labels = g.predict(X).tolist()
    gmm_means = g.means_.tolist()

    # MST (Ch 9.4) — minimum spanning tree over district-pair distances
    dist = squareform(pdist(X, metric="euclidean"))
    mst = minimum_spanning_tree(dist).toarray()
    edges = []
    for i in range(len(mst)):
        for j in range(len(mst)):
            if mst[i, j] > 0:
                edges.append({
                    "source": pcodes[i],
                    "target": pcodes[j],
                    "weight": float(mst[i, j]),
                })

    # DBSCAN-noise as clustering-based anomaly (Ch 10.5)
    db = DBSCAN(eps=0.6, min_samples=3).fit(X)
    noise_pcodes = [pcodes[i] for i, lab in enumerate(db.labels_) if lab == -1]

    return {
        "gmm": {
            "k_curve": gmm_k,
            "labels_k4": gmm_labels,
            "means_k4": gmm_means,
            "pcodes": pcodes,
            "features": feat_cols,
        },
        "mst": {"pcodes": pcodes, "edges": edges},
        "dbscan_noise_pcodes": noise_pcodes,
    }


# ---------------------------------------------------------------------------
# Entry point — append to existing viz_payload.json
# ---------------------------------------------------------------------------
def run() -> Dict[str, Any]:
    master_p = DATA_PROC / "master_panel.parquet"
    pca_p = DATA_PROC / "panel_pca.parquet"
    if not master_p.exists():
        raise FileNotFoundError(f"missing {master_p}")

    master = pd.read_parquet(master_p)
    pca_df = pd.read_parquet(pca_p) if pca_p.exists() else master

    extra = {
        "classifiers": train_extra_classifiers(pca_df),
        "clustering": train_extra_clustering(master),
        "generated_at": pd.Timestamp.now().isoformat(),
    }

    # Return extra_models dict without writing to file
    # The orchestrator (train_pipeline.py) will merge this into viz_payload.json
    logger.info(f"extra_models prepared: classifiers={len(extra.get('classifiers', {}))}, clustering={bool(extra.get('clustering'))}")
    return extra


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run()
    print("extra_models trained and saved")
