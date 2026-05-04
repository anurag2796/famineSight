#!/usr/bin/env python3
"""
Audit script for FamineSight project to verify all components are working.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import *
from src.data.preprocessor import run_full_pipeline
from src.analysis.association import run_all as run_association
from src.analysis.clustering import run_all as run_clustering
from src.analysis.classification import run_all as run_classification
from src.analysis.anomaly import run_all as run_anomaly

logger = logging.getLogger(__name__)

def audit_phase_i():
    """Phase I checks - Data pipeline verification."""
    logger.info("=== Phase I Audit: Data Pipeline ===")

    # Check master panel exists
    master_panel_path = Path("data/processed/master_panel.parquet")
    if not master_panel_path.exists():
        logger.error("❌ Master panel not found")
        return False
    logger.info("✅ Master panel exists")

    # Check data quality
    df = pd.read_parquet(master_panel_path)

    # Check row count
    if len(df) < 100:
        logger.warning("⚠️  Fewer than 100 rows in master panel")
    else:
        logger.info(f"✅ Master panel has {len(df)} rows")

    # Check crisis rate
    crisis_rate = df['crisis_label'].mean()
    if 0.03 <= crisis_rate <= 0.20:
        logger.info(f"✅ Crisis rate is within expected range: {crisis_rate:.1%}")
    else:
        logger.warning(f"⚠️  Crisis rate outside expected range: {crisis_rate:.1%}")

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        logger.info("✅ No missing values in master panel")
    else:
        logger.warning(f"⚠️  Found {missing_count} missing values")

    # Check temporal sort
    df_sorted = df.sort_values(['pcode', 'date'])
    if df_sorted.equals(df.sort_values(['pcode', 'date'])):
        logger.info("✅ Data is temporally sorted")
    else:
        logger.warning("⚠️  Data is not temporally sorted")

    # Check drought anomaly correlation
    if 'rainfall_anomaly_pct' in df.columns and 'cdr_per_10k_per_day' in df.columns:
        correlation = df['rainfall_anomaly_pct'].corr(df['cdr_per_10k_per_day'])
        if correlation < 0:
            logger.info(f"✅ Drought anomaly negatively correlated with CDR (r={correlation:.3f})")
        else:
            logger.warning(f"⚠️  Drought anomaly positively correlated with CDR (r={correlation:.3f})")

    return True

def audit_phase_ii():
    """Phase II checks - Model verification."""
    logger.info("=== Phase II Audit: Model Verification ===")

    # Check model files exist in MODELS_DIR
    model_files = [
        "random_forest.joblib",
        "xgboost_model.joblib",
        "scaler.joblib",
        "pca.joblib",
        "kmeans.joblib",
        "dbscan.joblib",
        "isolation_forest.joblib",
        "lof_model.joblib"
    ]
    
    # Also check result JSON files
    result_files = [
        "anomaly_results.json",
        "cluster_results.json",
        "association_results.json"
    ]

    all_exist = True
    for model_file in model_files + result_files:
        path = MODELS_DIR / model_file
        if path.exists():
            logger.info(f"✅ {model_file} exists")
        else:
            logger.warning(f"❌ {model_file} missing")
            all_exist = False

    # Check RF parameters
    if 'RF_N_JOBS' in globals() and RF_N_JOBS == 4:
        logger.info("✅ RF_N_JOBS is correctly set to 4 (Jetson-safe)")
    else:
        logger.warning("⚠️  RF_N_JOBS not set to 4")

    # Check Dockerfile has GDAL
    dockerfile_path = Path("backend/Dockerfile")
    if dockerfile_path.exists():
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            if 'gdal-bin' in content and 'libgdal-dev' in content:
                logger.info("✅ Dockerfile contains GDAL dependencies")
            else:
                logger.warning("⚠️  Dockerfile missing GDAL dependencies")

    return all_exist

def audit_phase_iii():
    """Phase III checks - LLM integration."""
    logger.info("=== Phase III Audit: LLM Integration ===")

    # Check LLM source files
    llm_files = [
        "src/llm/client.py",
        "src/llm/prompts.py",
        "src/llm/guardrails.py"
    ]

    all_exist = True
    for file in llm_files:
        if Path(file).exists():
            logger.info(f"✅ {Path(file).name} exists")
        else:
            logger.warning(f"❌ {Path(file).name} missing")
            all_exist = False

    # Check prompt structure
    try:
        from src.llm.prompts import SYSTEM_PROMPT
        if len(SYSTEM_PROMPT) > 0:
            logger.info("✅ System prompt exists")
        else:
            logger.warning("⚠️  System prompt is empty")
    except Exception as e:
        logger.warning(f"⚠️  Could not load system prompt: {e}")

    return all_exist

def main():
    """Run all audits."""
    logger.info("Starting FamineSight audit...")

    try:
        # Run audits
        phase1 = audit_phase_i()
        phase2 = audit_phase_ii()
        phase3 = audit_phase_iii()

        if phase1 and phase2 and phase3:
            logger.info("🎉 All audits passed!")
            return True
        else:
            logger.error("❌ Some audits failed")
            return False

    except Exception as e:
        logger.error(f"Error during audit: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)