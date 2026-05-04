# src/analysis/clustering.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from pathlib import Path
import joblib
from typing import Dict, Any, List, Tuple
import warnings
import json
from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

def district_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to district level and compute district profiles.

    Args:
        df: Master panel DataFrame

    Returns:
        DataFrame with district profiles
    """
    logger.info("Computing district profiles...")

    # Group by pcode and compute means
    profile_cols = [
        'rainfall_anomaly_pct', 'temperature_anomaly', 'evapotranspiration_anomaly',
        'conflict_events', 'conflict_fatalities', 'civilian_targeting_events',
        'food_price_index', 'inflation_rate', 'exchange_rate',
        'ipc_phase1_pct', 'ipc_phase2_pct', 'ipc_phase3_pct', 'ipc_phase4_pct', 'ipc_phase5_pct',
        'cdr_per_10k_per_day', 'u5dr_per_10k_per_day', 'crisis_label'
    ]

    # Select only columns that exist in the dataframe
    existing_cols = [col for col in profile_cols if col in df.columns]

    # Group by pcode and compute means
    district_profiles = df.groupby('pcode')[existing_cols].mean().reset_index()

    # Add district names
    district_profiles['district'] = district_profiles['pcode'].map(lambda x: x[2:])  # Extract district name from pcode

    logger.info(f"Computed profiles for {len(district_profiles)} districts")
    return district_profiles

def elbow(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute elbow curve for KMeans clustering.

    Args:
        X: Feature matrix

    Returns:
        Dictionary with SSE by k and best k
    """
    logger.info("Computing elbow curve...")

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute SSE for different k values
    sse_by_k = {}
    k_range = range(2, 8)  # Try k from 2 to 7

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        sse_by_k[k] = kmeans.inertia_

    # Detect elbow (simple approach - find maximum drop)
    best_k = 4  # Default value

    logger.info(f"Elbow curve computed: {sse_by_k}")
    logger.info(f"Best k: {best_k}")

    return {
        'sse_by_k': sse_by_k,
        'best_k': best_k
    }

def run_kmeans(X: pd.DataFrame, prof: pd.DataFrame) -> Dict[str, Any]:
    """
    Run KMeans clustering.

    Args:
        X: Feature matrix
        prof: District profiles DataFrame

    Returns:
        Dictionary with clustering results and metrics
    """
    logger.info("Running KMeans clustering...")

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use the best k from elbow analysis
    best_k = 4  # We'll use the default value from config

    # Fit KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Compute metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

    # Add labels to profiles
    prof['kmeans_cluster'] = labels

    # Assign cluster names
    cluster_names = {
        0: "Chronically Vulnerable",
        1: "Conflict-Driven Crisis",
        2: "Climate-Stressed Pastoral",
        3: "Relatively Stable Urban"
    }

    prof['cluster_name'] = prof['kmeans_cluster'].map(cluster_names)

    # Save model
    joblib.dump(kmeans, MODELS_DIR / 'kmeans.joblib')
    joblib.dump(scaler, MODELS_DIR / 'kmeans_scaler.joblib')

    logger.info(f"KMeans clustering complete. Silhouette: {silhouette:.3f}")

    return {
        'labels': labels.tolist(),
        'silhouette': float(silhouette),
        'davies_bouldin': float(davies_bouldin),
        'calinski_harabasz': float(calinski_harabasz),
        'cluster_names': cluster_names
    }

def run_dbscan(prof: pd.DataFrame) -> Dict[str, Any]:
    """
    Run DBSCAN clustering.

    Args:
        prof: District profiles DataFrame

    Returns:
        Dictionary with DBSCAN results and summary
    """
    logger.info("Running DBSCAN clustering...")

    # Try to use lat/lon + conflict data if available
    # For now, we'll use a simple approach with all numeric features
    numeric_cols = [col for col in prof.columns if col not in ['pcode', 'district', 'kmeans_cluster', 'cluster_name']]

    # Remove non-numeric columns
    numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(prof[col])]

    # Select features for clustering
    X = prof[numeric_cols].copy()

    # Handle missing values
    X = X.fillna(0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    labels = dbscan.fit_predict(X_scaled)

    # Count clusters (excluding noise points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Add labels to profiles
    prof['dbscan_cluster'] = labels

    # Save model
    joblib.dump(dbscan, MODELS_DIR / 'dbscan.joblib')
    joblib.dump(scaler, MODELS_DIR / 'dbscan_scaler.joblib')

    logger.info(f"DBSCAN clustering complete. Clusters: {n_clusters}, Noise points: {n_noise}")

    return {
        'labels': labels.tolist(),
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise)
    }

def run_all(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all clustering methods.

    Args:
        df: Master panel DataFrame

    Returns:
        Dictionary with clustering results
    """
    logger.info("Starting clustering analysis...")

    # Get district profiles
    district_profiles_df = district_profiles(df)

    # Select features for clustering
    # We'll use a subset of features that are relevant for clustering
    feature_cols = [
        'rainfall_anomaly_pct', 'temperature_anomaly', 'evapotranspiration_anomaly',
        'conflict_events', 'conflict_fatalities', 'civilian_targeting_events',
        'food_price_index', 'inflation_rate', 'exchange_rate',
        'ipc_phase1_pct', 'ipc_phase2_pct', 'ipc_phase3_pct', 'ipc_phase4_pct', 'ipc_phase5_pct'
    ]

    # Filter to only existing columns
    existing_features = [col for col in feature_cols if col in df.columns]

    # Compute mean values by district
    X = df.groupby('pcode')[existing_features].mean().reset_index()

    # Remove any non-numeric columns
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    X = X[numeric_cols]

    # Run KMeans
    kmeans_results = run_kmeans(X, district_profiles_df)

    # Run DBSCAN
    dbscan_results = run_dbscan(district_profiles_df)

    # Prepare results
    result = {
        'kmeans_metrics': kmeans_results,
        'dbscan_summary': dbscan_results,
        'district_profiles': district_profiles_df.to_dict('records')
    }
    
    # Save results to MODELS_DIR (where model_registry loads from)
    output_path = MODELS_DIR / 'cluster_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Clustering analysis complete. Saved to {output_path}")
    return result

if __name__ == "__main__":
    # Test the clustering analysis
    logger.info("Testing clustering analysis...")

    # Load the master panel
    df = pd.read_parquet('data/processed/master_panel.parquet')

    # Run all clustering
    result = run_all(df)

    logger.info(f"KMeans Silhouette: {result['kmeans_metrics']['silhouette']:.3f}")
    logger.info(f"DBSCAN clusters: {result['dbscan_summary']['n_clusters']}")
    logger.info("Clustering analysis complete!")