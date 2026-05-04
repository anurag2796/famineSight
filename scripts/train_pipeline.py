#!/usr/bin/env python3
"""
Master training script for FamineSight pipeline.
This script runs all phases of the data mining pipeline in order.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import run_full_pipeline
from src.analysis.classification import run_all as run_classification
from src.analysis.anomaly import run_all as run_anomaly
from src.analysis.clustering import run_all as run_clustering
from src.analysis.association import run_all as run_association
from src.analysis.viz_payload import build_payload as build_viz_payload
from src.analysis.extra_models import run as run_extra_models
from data.synthetic.generate_synthetic import main as generate_synthetic
from scripts.fetch_data import fetch_all_data

# Set up logging — use absolute path so this works from any CWD (e.g. Docker)
_log_dir = PROJECT_ROOT / 'logs'
_log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(_log_dir / 'train_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def banner(step, name):
    """Print a formatted banner for each step."""
    print(f"\n{'='*50}")
    print(f"STEP {step}: {name}")
    print(f"{'='*50}")

def main(synthetic_only=False):
    """Run the complete training pipeline."""
    logger.info("Starting FamineSight training pipeline...")

    # Step 1: Generate synthetic data (only when forced or raw data is missing)
    banner(1, "Generate Synthetic Data")
    raw_data_present = any([
        (PROJECT_ROOT / "data" / "raw" / "acled" / "somalia_acled.csv").exists(),
        (PROJECT_ROOT / "data" / "raw" / "ipc"  / "ipc_phases.csv").exists(),
    ])

    if synthetic_only or not raw_data_present:
        try:
            generate_synthetic()
            logger.info("Synthetic data generation complete")
        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            raise
    else:
        logger.info("Real raw data already present — skipping synthetic generation")

    # Step 2: Fetch data (or use synthetic if forced)
    banner(2, "Fetch Data")
    try:
        fetch_all_data(force_synthetic=synthetic_only)
        logger.info("Data fetching complete")
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

    # Step 3: Run preprocessing pipeline
    banner(3, "Run Preprocessing Pipeline")
    try:
        df = run_full_pipeline(save=True)
        logger.info(f"Preprocessing complete. Panel shape: {df.shape}")
        logger.info(f"Crisis rate: {df['crisis_label'].mean():.1%}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
    except Exception as e:
        logger.error(f"Failed to run preprocessing: {e}")
        raise

    # Step 4: Run association rule mining
    banner(4, "Run Association Rule Mining")
    try:
        association_results = run_association(df)
        logger.info(f"Association rules complete. FP-Growth: {association_results['stats']['fp_n_rules']}, Apriori: {association_results['stats']['apriori_n_rules']}")
    except Exception as e:
        logger.error(f"Failed to run association rule mining: {e}")
        raise

    # Step 5: Run clustering
    banner(5, "Run Clustering")
    try:
        clustering_results = run_clustering(df)
        logger.info(f"Clustering complete. KMeans silhouette: {clustering_results['kmeans_metrics']['silhouette']:.3f}")
        logger.info(f"DBSCAN clusters: {clustering_results['dbscan_summary']['n_clusters']}")
    except Exception as e:
        logger.error(f"Failed to run clustering: {e}")
        raise

    # Step 6: Run classification
    banner(6, "Run Classification")
    try:
        classification_results = run_classification(df)
        logger.info(f"Classification complete. RF ROC-AUC: {classification_results['random_forest']['roc_auc']:.3f}")
        logger.info(f"RF Recall: {classification_results['random_forest']['recall']:.3f}")
    except Exception as e:
        logger.error(f"Failed to run classification: {e}")
        raise

    # Step 7: Run anomaly detection
    banner(7, "Run Anomaly Detection")
    try:
        anomaly_results = run_anomaly(df)
        logger.info(f"Anomaly detection complete. Total anomalies: {anomaly_results['total_anomalies']}")
        logger.info(f"Critical alerts: {anomaly_results['critical_count']}")
    except Exception as e:
        logger.error(f"Failed to run anomaly detection: {e}")
        raise

    # Step 8: Phase-B syllabus models (kNN, NB, SVM, MLP, GMM, MST)
    banner(8, "Train Phase-B syllabus models")
    extra_models_data = None
    try:
        extra_models_data = run_extra_models()
        logger.info(f"Phase-B models trained: {len(extra_models_data.get('classifiers', {}))} classifiers, clustering={'enabled' if extra_models_data.get('clustering') else 'skipped'}")
    except Exception as e:
        logger.warning(f"Phase-B models step failed (non-fatal): {e}")
        extra_models_data = None

    # Step 9: Build chart-ready payload for the dashboard (merged with Phase-B results)
    banner(9, "Build viz payload for frontend")
    try:
        build_viz_payload(extra_models=extra_models_data)
        logger.info("viz_payload.json built")
        
        # Integrity check: verify extra_models persists in final payload
        if extra_models_data:
            import json
            payload_path = PROJECT_ROOT / "models" / "viz_payload.json"
            with open(payload_path) as f:
                final_payload = json.load(f)
            if "extra_models" in final_payload:
                logger.info("SUCCESS: extra_models successfully merged into final viz_payload.json")
            else:
                logger.error("CRITICAL: extra_models was trained but NOT in final payload!")
    except Exception as e:
        logger.warning(f"viz payload step failed (non-fatal): {e}")

    logger.info("Training pipeline complete!")

if __name__ == "__main__":
    # Check if --synthetic flag is provided
    synthetic_only = "--synthetic" in sys.argv

    main(synthetic_only)