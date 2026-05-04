# src/analysis/anomaly.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import logging
from typing import Dict, List, Any
import warnings
from pathlib import Path
from src.config import (
    ISOFOREST_CONTAMINATION, LOF_N_NEIGHBORS, ZSCORE_THRESHOLD, MODELS_DIR
)
import json

logger = logging.getLogger(__name__)

def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest anomaly detection.

    Args:
        df: Master panel DataFrame

    Returns:
        DataFrame with anomaly scores and flags
    """
    logger.info("Running Isolation Forest...")

    # Select numeric features for anomaly detection
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    # Filter to only relevant features
    relevant_features = [
        'rainfall_anomaly_pct', 'conflict_fatalities', 'food_price_index',
        'cdr_per_10k_per_day', 'u5dr_per_10k_per_day'
    ]

    # Keep only features that exist in the dataframe
    selected_features = [col for col in relevant_features if col in df.columns]

    if not selected_features:
        logger.warning("No relevant features for Isolation Forest")
        return df

    # Extract features
    X = df[selected_features].copy()

    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Run Isolation Forest
    iso_forest = IsolationForest(
        contamination=ISOFOREST_CONTAMINATION,
        random_state=42,
        n_estimators=100
    )

    # Fit and predict
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.decision_function(X_scaled)

    # Add results to dataframe
    df['isolation_forest_score'] = anomaly_scores
    df['isolation_forest_anomaly'] = anomaly_labels == -1  # -1 indicates anomaly

    # Save model
    import joblib
    joblib.dump(iso_forest, MODELS_DIR / 'isolation_forest.joblib')
    joblib.dump(scaler, MODELS_DIR / 'isolation_forest_scaler.joblib')

    logger.info(f"Isolation Forest complete. Found {df['isolation_forest_anomaly'].sum()} anomalies")
    return df

def run_lof(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Local Outlier Factor anomaly detection.

    Args:
        df: Master panel DataFrame

    Returns:
        DataFrame with LOF scores and flags
    """
    logger.info("Running Local Outlier Factor...")

    # Select numeric features for anomaly detection
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    # Filter to only relevant features
    relevant_features = [
        'rainfall_anomaly_pct', 'conflict_fatalities', 'food_price_index',
        'cdr_per_10k_per_day', 'u5dr_per_10k_per_day'
    ]

    # Keep only features that exist in the dataframe
    selected_features = [col for col in relevant_features if col in df.columns]

    if not selected_features:
        logger.warning("No relevant features for LOF")
        return df

    # Extract features
    X = df[selected_features].copy()

    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Run LOF with novelty=True for inference support
    lof = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, novelty=True)

    # novelty=True: use fit() + score_samples() + predict()
    # negative_outlier_factor_ is NOT populated in novelty mode — use score_samples() instead
    lof.fit(X_scaled)
    lof_scores = lof.score_samples(X_scaled)  # higher = more normal, negative = outlier
    anomaly_labels = lof.predict(X_scaled)    # -1 = anomaly, +1 = normal

    # Add results to dataframe
    df['lof_anomaly'] = anomaly_labels == -1  # -1 indicates anomaly
    df['lof_score'] = lof_scores

    # Save model
    import joblib
    joblib.dump(lof, MODELS_DIR / 'lof_model.joblib')
    joblib.dump(scaler, MODELS_DIR / 'lof_scaler.joblib')

    logger.info(f"LOF complete. Found {df['lof_anomaly'].sum()} anomalies")
    return df

def run_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Z-score based anomaly detection.

    Args:
        df: Master panel DataFrame

    Returns:
        DataFrame with Z-score and spike flags
    """
    logger.info("Running Z-score analysis...")

    # Select numeric features for anomaly detection
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    # Filter to only relevant features
    relevant_features = [
        'food_price_index', 'cdr_per_10k_per_day', 'conflict_fatalities'
    ]

    # Keep only features that exist in the dataframe
    selected_features = [col for col in relevant_features if col in df.columns]

    if not selected_features:
        logger.warning("No relevant features for Z-score analysis")
        return df

    # Calculate Z-scores for each feature by district
    for feature in selected_features:
        # Group by pcode and calculate z-scores
        if feature in df.columns:
            # Calculate mean and std by district
            district_means = df.groupby('pcode')[feature].mean()
            district_stds = df.groupby('pcode')[feature].std()

            # Add to dataframe
            df[f'{feature}_zscore'] = df.groupby('pcode')[feature].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)

            # Flag spikes
            df[f'{feature}_spike'] = df[f'{feature}_zscore'].abs() > ZSCORE_THRESHOLD

    logger.info("Z-score analysis complete")
    return df

def generate_alerts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate alerts based on anomaly detection results.

    Args:
        df: DataFrame with anomaly flags

    Returns:
        List of alert dictionaries
    """
    logger.info("Generating alerts...")

    # Find rows where both isolation forest and LOF detect anomalies
    anomaly_mask = (df['isolation_forest_anomaly'] & df['lof_anomaly'])

    # Filter to get anomaly rows
    anomaly_rows = df[anomaly_mask].copy()

    # Create alert list
    alerts = []

    for _, row in anomaly_rows.iterrows():
        # Determine severity
        severity = "CRITICAL" if row.get('crisis_label', 0) == 1 else "WARNING"

        alert = {
            'date': row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
            'district': row.get('district', 'Unknown'),
            'pcode': row.get('pcode', 'Unknown'),
            'severity': severity,
            'cdr': row.get('cdr_per_10k_per_day', 0),
            'anomaly_flags': {
                'isolation_forest': row.get('isolation_forest_anomaly', False),
                'lof': row.get('lof_anomaly', False),
                'zscore': row.get('food_price_index_spike', False) or
                          row.get('cdr_per_10k_per_day_spike', False) or
                          row.get('conflict_fatalities_spike', False)
            }
        }

        alerts.append(alert)

    # Sort by severity (critical first) and date
    alerts.sort(key=lambda x: (x['severity'] != 'CRITICAL', x['date']))

    logger.info(f"Generated {len(alerts)} alerts")
    return alerts

def run_all(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all anomaly detection methods.

    Args:
        df: Master panel DataFrame

    Returns:
        Dictionary with all anomaly detection results
    """
    logger.info("Starting anomaly detection...")

    # Run all anomaly detection methods
    df_with_iso = run_isolation_forest(df.copy())
    df_with_lof = run_lof(df_with_iso)
    df_with_zscore = run_zscore(df_with_lof)

    # Generate alerts
    alerts = generate_alerts(df_with_zscore)

    # Calculate statistics
    total_anomalies = df_with_zscore['isolation_forest_anomaly'].sum() + df_with_zscore['lof_anomaly'].sum()
    critical_count = sum(1 for alert in alerts if alert['severity'] == 'CRITICAL')
    anomaly_rate = total_anomalies / len(df_with_zscore) if len(df_with_zscore) > 0 else 0

    # Prepare results
    result = {
        'alerts': alerts[:100],  # Top 100 alerts
        'total_anomalies': int(total_anomalies),
        'anomaly_rate': float(anomaly_rate),
        'critical_count': int(critical_count)
    }
    
    # Save results to MODELS_DIR (where model_registry loads from)
    output_path = MODELS_DIR / 'anomaly_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Anomaly detection complete. Saved to {output_path}")
    return result

if __name__ == "__main__":
    # Test the anomaly detection
    logger.info("Testing anomaly detection...")

    # Load the master panel
    df = pd.read_parquet('data/processed/master_panel.parquet')

    # Run all anomaly detection
    result = run_all(df)

    print(f'Total alerts: {result["total_anomalies"]}')
    print(f'Critical: {result["critical_count"]}')
    print(f'Rate: {result["anomaly_rate"]:.1%}')

    print('Anomaly detection test completed!')