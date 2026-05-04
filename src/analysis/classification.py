# src/analysis/classification.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
from src.config import (
    RF_N_ESTIMATORS, XGB_SCALE_POS_WEIGHT, SMOTE_K_NEIGHBORS,
    RF_N_JOBS, XGB_DEVICE, RANDOM_STATE, MODELS_DIR
)
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

# Current-period IPC columns that directly encode the crisis_label definition.
# Excluding them ("lead_only" mode) gives the honest forecasting problem where
# models must predict crises from lagged signals rather than re-learning the label.
CURRENT_IPC_COLS = [
    'ipc_phase1_pct', 'ipc_phase2_pct', 'ipc_phase3_pct',
    'ipc_phase4_pct', 'ipc_phase5_pct',
]

def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally using percentile cutoffs (70% train / 15% val / 15% test).

    Using percentile-based cutoffs instead of hard year thresholds ensures the
    training set always captures enough positive (crisis) labels even when crises
    are concentrated in recent years.

    Args:
        df: Master panel DataFrame

    Returns:
        Tuple of (train, val, test) DataFrames
    """
    logger.info("Performing temporal split (percentile-based)...")

    # Ensure chronological order by time then pcode so ties are deterministic
    df_sorted = df.sort_values(['date', 'pcode']).reset_index(drop=True)

    # Derive cutoff dates at the 70th and 85th percentile of unique sorted dates
    unique_dates = df_sorted['date'].sort_values().unique()
    n = len(unique_dates)
    train_cut = unique_dates[int(n * 0.70)]
    val_cut   = unique_dates[int(n * 0.85)]

    train_df = df_sorted[df_sorted['date'] <  train_cut]
    val_df   = df_sorted[(df_sorted['date'] >= train_cut) & (df_sorted['date'] < val_cut)]
    test_df  = df_sorted[df_sorted['date'] >= val_cut]

    # Re-sort within each split
    train_df = train_df.sort_values(['pcode', 'date'])
    val_df   = val_df.sort_values(['pcode', 'date'])
    test_df  = test_df.sort_values(['pcode', 'date'])

    logger.info(
        f"Temporal split — Train: {len(train_df)} rows "
        f"({train_df['crisis_label'].sum()} crisis, cutoff {train_cut.date()}), "
        f"Val: {len(val_df)} rows ({val_df['crisis_label'].sum()} crisis, cutoff {val_cut.date()}), "
        f"Test: {len(test_df)} rows ({test_df['crisis_label'].sum()} crisis)"
    )

    return train_df, val_df, test_df

def get_features(df: pd.DataFrame, feature_mode: str = "full") -> Tuple[pd.DataFrame, list]:
    """
    Extract numeric feature columns from DataFrame.

    Args:
        df: DataFrame with features
        feature_mode: "full" (all features) or "lead_only" (exclude current-period
                      IPC columns to avoid re-learning the label definition).

    Returns:
        Tuple of (features DataFrame, feature column names)
    """
    exclude = ['date', 'pcode', 'district', 'crisis_label',
               'cdr_per_10k_per_day', 'u5dr_per_10k_per_day']
    if feature_mode == "lead_only":
        exclude += CURRENT_IPC_COLS

    feature_cols = [
        col for col in df.columns
        if (pd.api.types.is_numeric_dtype(df[col]) and col not in exclude)
    ]

    X = df[feature_cols].copy()
    X = X.dropna(axis=1, how='all')
    feature_cols = X.columns.tolist()

    logger.info(f"Extracted {len(feature_cols)} features (mode={feature_mode})")
    return X, feature_cols

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> Dict[str, Any]:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        name: Model name for logging

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {name} model...")

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # AUC and average precision (only if probabilities available)
    roc_auc = 0
    avg_precision = 0

    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not compute AUC/precision metrics: {e}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Count false negatives (missed crises - lives lost)
    # cm[1, 0] is the number of actual crises predicted as no crisis (false negatives)
    false_negatives = cm[1, 0] if cm.shape == (2, 2) else 0

    # Log results
    logger.info(f"{name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    logger.info(f"{name} - ROC-AUC: {roc_auc:.3f}, Avg Precision: {avg_precision:.3f}")
    logger.info(f"{name} - False Negatives (missed crises): {false_negatives}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'false_negatives': false_negatives
    }

def train_rf(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Trained Random Forest model
    """
    logger.info("Training Random Forest...")

    # Handle class imbalance — prefer SMOTE, fall back to class_weight='balanced'
    minority_class_count = int(y_train.sum())
    smote_applied = False

    # SMOTE needs at least k_neighbors+1 samples in the minority class
    if minority_class_count >= 2:
        k_neighbors = min(SMOTE_K_NEIGHBORS, minority_class_count - 1)
        k_neighbors = max(1, k_neighbors)

        # Impute any NaNs before SMOTE
        if X_train.isnull().any().any():
            logger.warning("NaN values detected in features, imputing with KNN...")
            imputer = KNNImputer(n_neighbors=5)
            X_train_imputed = pd.DataFrame(
                imputer.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            X_train_imputed = X_train

        smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
        smote_applied = True
        logger.info(f"SMOTE: Resampled from {len(X_train)} to {len(X_train_resampled)} samples (k={k_neighbors})")
    else:
        logger.warning(
            f"Only {minority_class_count} minority-class sample(s) — skipping SMOTE, "
            "relying on class_weight='balanced' instead."
        )
        X_train_resampled, y_train_resampled = X_train, y_train

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE,
        oob_score=True,
        class_weight='balanced'
    )

    rf.fit(X_train_resampled, y_train_resampled)

    # Save model
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    logger.info("Random Forest training complete")
    return rf

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets

    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost...")

    # Handle class imbalance with scale_pos_weight
    n_negative = len(y_train) - y_train.sum()
    n_positive = y_train.sum()

    if n_positive > 0:
        scale_pos_weight = n_negative / n_positive
    else:
        scale_pos_weight = 1.0

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        tree_method="hist",  # CPU-optimized histogram method for ARM64
        device=XGB_DEVICE,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )

    # Fit without early stopping for compatibility
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Save model
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.joblib")

    logger.info("XGBoost training complete")
    return xgb_model

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Train a simple Decision Tree Classifier for interpretability.
    """
    logger.info("Training Decision Tree...")
    
    # Train simple interpretable tree (max_depth=4)
    dt_model = DecisionTreeClassifier(
        max_depth=4,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    
    dt_model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(dt_model, MODELS_DIR / "decision_tree.joblib")
    
    logger.info("Decision Tree training complete")
    return dt_model

def train_decision_tree_optimized(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    feature_cols: list,
) -> Tuple[DecisionTreeClassifier, float, str]:
    """
    Train an optimized Decision Tree on lead_only features.

    Uses SMOTE for class imbalance, GridSearchCV for hyperparameter tuning,
    and threshold sweeping on the validation set to maximise recall while
    keeping precision >= 0.50 (humanitarian priority: never miss a crisis).

    Returns:
        (best_estimator, optimal_threshold, text_rules)
    """
    logger.info("Training optimized Decision Tree (GridSearchCV + threshold tuning)...")

    # SMOTE — same guard logic as train_rf
    minority_count = int(y_train.sum())
    if minority_count >= 2:
        k = max(1, min(SMOTE_K_NEIGHBORS, minority_count - 1))
        X_imp = X_train.copy()
        if X_imp.isnull().any().any():
            _imp = KNNImputer(n_neighbors=5)
            X_imp = pd.DataFrame(_imp.fit_transform(X_imp), columns=X_imp.columns)
        smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_imp, y_train)
        logger.info(f"SMOTE: {len(X_train)} → {len(X_res)} samples")
    else:
        logger.warning(f"Only {minority_count} minority samples — skipping SMOTE")
        X_res = X_train.fillna(0)
        y_res = y_train

    # GridSearchCV — StratifiedKFold preserves class ratio across folds
    param_grid = {
        'max_depth':          [3, 4, 5, 6, 8, None],
        'min_samples_split':  [2, 5, 10, 20],
        'min_samples_leaf':   [1, 2, 4],
        'criterion':          ['gini', 'entropy'],
        'class_weight':       ['balanced', None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring='f1', n_jobs=RF_N_JOBS, refit=True,
    )
    gs.fit(X_res, y_res)
    best_dt = gs.best_estimator_
    logger.info(f"Best params: {gs.best_params_}  CV F1: {gs.best_score_:.3f}")

    # Threshold sweep on validation set — maximise recall s.t. precision >= 0.50
    X_val_arr = X_val.fillna(0).values
    proba_val = best_dt.predict_proba(X_val_arr)[:, 1]

    best_threshold = 0.5
    best_recall = 0.0
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred_t = (proba_val >= thresh).astype(int)
        prec_t = precision_score(y_val, y_pred_t, zero_division=0)
        rec_t  = recall_score(y_val, y_pred_t, zero_division=0)
        if prec_t >= 0.50 and rec_t > best_recall:
            best_recall    = rec_t
            best_threshold = float(thresh)

    logger.info(f"Optimal threshold: {best_threshold:.2f}  (val recall={best_recall:.3f})")

    # Attach metadata so it survives joblib round-trip
    best_dt._famine_threshold    = best_threshold
    best_dt._famine_feature_cols = feature_cols

    joblib.dump(best_dt, MODELS_DIR / "decision_tree_optimized.joblib")
    dt_rules = export_text(best_dt, feature_names=feature_cols)
    return best_dt, best_threshold, dt_rules


def compare_models(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    dt_opt_model=None, dt_opt_threshold: float = 0.5,
    dt_opt_feature_cols: list = None,
) -> list:
    """
    Evaluate RF, XGBoost, basic DT, and optimized DT on both feature modes.

    Lightweight in-memory RF/XGB variants (50 estimators) are used so the
    saved primary models are not overwritten.

    Returns a list of per-(model, feature_mode) result dicts.
    """
    logger.info("Running model comparison (full vs lead_only features)...")
    comparison = []

    for mode in ["full", "lead_only"]:
        X_tr, fcols = get_features(train_df, feature_mode=mode)
        X_v,  _     = get_features(val_df,   feature_mode=mode)
        X_te, _     = get_features(test_df,  feature_mode=mode)

        imp = KNNImputer(n_neighbors=5)
        X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=fcols)
        X_v_imp  = pd.DataFrame(imp.transform(X_v),      columns=fcols)
        X_te_imp = pd.DataFrame(imp.transform(X_te),     columns=fcols)

        spw = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

        candidates = [
            ("RandomForest", RandomForestClassifier(
                n_estimators=50, n_jobs=RF_N_JOBS,
                random_state=RANDOM_STATE, class_weight='balanced',
            )),
            ("XGBoost", xgb.XGBClassifier(
                n_estimators=50, tree_method="hist", device=XGB_DEVICE,
                scale_pos_weight=spw, random_state=RANDOM_STATE,
                eval_metric='logloss', verbosity=0,
            )),
            ("DT_basic", DecisionTreeClassifier(
                max_depth=4, class_weight='balanced', random_state=RANDOM_STATE,
            )),
        ]

        for model_name, mdl in candidates:
            try:
                if isinstance(mdl, xgb.XGBClassifier):
                    mdl.fit(X_tr_imp, y_train,
                            eval_set=[(X_v_imp, y_val)], verbose=False)
                else:
                    mdl.fit(X_tr_imp, y_train)

                # Tune threshold on validation set to maximise F1
                proba_v = mdl.predict_proba(X_v_imp.values)[:, 1]
                best_thresh, best_f1 = 0.5, 0.0
                for thr in np.arange(0.05, 0.95, 0.05):
                    p_t = (proba_v >= thr).astype(int)
                    f = f1_score(y_val, p_t, zero_division=0)
                    if f > best_f1:
                        best_f1, best_thresh = f, float(thr)

                proba_te = mdl.predict_proba(X_te_imp.values)[:, 1]
                y_pred_te = (proba_te >= best_thresh).astype(int)
                cm = confusion_matrix(y_test, y_pred_te)
                fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
                try:
                    auc = float(roc_auc_score(y_test, proba_te))
                except Exception:
                    auc = 0.0
                comparison.append({
                    'model': model_name, 'feature_mode': mode,
                    'precision': float(precision_score(y_test, y_pred_te, zero_division=0)),
                    'recall':    float(recall_score(y_test,    y_pred_te, zero_division=0)),
                    'f1':        float(f1_score(y_test,        y_pred_te, zero_division=0)),
                    'roc_auc': auc,
                    'false_negatives': fn,
                    'confusion_matrix': cm.tolist(),
                    'threshold': best_thresh,
                })
            except Exception as exc:
                logger.warning(f"compare_models {model_name}/{mode} failed: {exc}")

        # Optimized DT is only meaningful on lead_only (that's what it was trained on)
        if mode == "lead_only" and dt_opt_model is not None and dt_opt_feature_cols is not None:
            try:
                shared = [c for c in dt_opt_feature_cols if c in X_te_imp.columns]
                X_te_opt = X_te_imp[shared].fillna(0).values
                proba = dt_opt_model.predict_proba(X_te_opt)[:, 1]
                y_pred = (proba >= dt_opt_threshold).astype(int)
                cm = confusion_matrix(y_test, y_pred)
                fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
                try:
                    auc = float(roc_auc_score(y_test, proba))
                except Exception:
                    auc = 0.0
                comparison.append({
                    'model': 'DT_optimized', 'feature_mode': 'lead_only',
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall':    float(recall_score(y_test,    y_pred, zero_division=0)),
                    'f1':        float(f1_score(y_test,        y_pred, zero_division=0)),
                    'roc_auc': auc,
                    'false_negatives': fn,
                    'confusion_matrix': cm.tolist(),
                })
            except Exception as exc:
                logger.warning(f"compare_models DT_optimized failed: {exc}")

    logger.info(f"Model comparison complete: {len(comparison)} entries")
    return comparison


def shap_importance(rf_model, X_sample: pd.DataFrame, feature_cols: list) -> Dict[str, float]:
    """
    Compute SHAP feature importance.

    Args:
        rf_model: Trained Random Forest model
        X_sample: Sample features
        feature_cols: Feature column names

    Returns:
        Dictionary with feature importance values
    """
    try:
        import shap

        # Create explainer
        explainer = shap.TreeExplainer(rf_model)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Get mean absolute SHAP values for each feature
        shap_importance = np.abs(shap_values).mean(axis=0)

        # Create dictionary
        importance_dict = dict(zip(feature_cols, shap_importance))

        logger.info("SHAP importance computed")
        return importance_dict

    except Exception as e:
        logger.warning(f"Could not compute SHAP importance: {e}")
        return {}

def run_all(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete classification pipeline.

    Args:
        df: Master panel DataFrame

    Returns:
        Dictionary with all model results
    """
    logger.info("Starting classification pipeline...")

    # Perform temporal split
    train_df, val_df, test_df = temporal_split(df)

    # Extract features
    X_train, feature_cols = get_features(train_df)
    X_val, _ = get_features(val_df)
    X_test, _ = get_features(test_df)

    # Extract targets
    y_train = train_df['crisis_label']
    y_val = val_df['crisis_label']
    y_test = test_df['crisis_label']

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)

    # Handle missing values in training data
    if X_train.isnull().any().any():
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        logger.info(f"Imputed missing values in training data")
    else:
        X_train_imputed = X_train

    # Handle missing values in validation and test data
    if X_val.isnull().any().any():
        X_val_imputed = pd.DataFrame(
            imputer.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        logger.info(f"Imputed missing values in validation data")
    else:
        X_val_imputed = X_val

    if X_test.isnull().any().any():
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        logger.info(f"Imputed missing values in test data")
    else:
        X_test_imputed = X_test

    # The master panel is already scaled by preprocessor.py using MinMaxScaler.
    # Therefore, we skip re-scaling here to ensure inference exactly matches training.
    X_train_scaled = X_train_imputed
    X_val_scaled = X_val_imputed
    X_test_scaled = X_test_imputed

    # Train models
    logger.info("Training models...")

    rf_model  = train_rf(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
    dt_model  = train_decision_tree(X_train_scaled, y_train)

    # Optimized DT — trained on lead_only features (no current-period IPC)
    X_train_lead, lead_cols = get_features(train_df, feature_mode="lead_only")
    X_val_lead,   _         = get_features(val_df,   feature_mode="lead_only")
    lead_imputer = KNNImputer(n_neighbors=5)
    if X_train_lead.isnull().any().any():
        X_train_lead = pd.DataFrame(
            lead_imputer.fit_transform(X_train_lead), columns=lead_cols)
        X_val_lead = pd.DataFrame(
            lead_imputer.transform(X_val_lead), columns=lead_cols)
    dt_opt_model, dt_opt_threshold, dt_opt_rules = train_decision_tree_optimized(
        X_train_lead, y_train, X_val_lead, y_val, lead_cols,
    )

    # Evaluate all four models on the full feature test set
    rf_results  = evaluate(rf_model,  X_test_scaled, y_test, "Random Forest")
    xgb_results = evaluate(xgb_model, X_test_scaled, y_test, "XGBoost")
    dt_results  = evaluate(dt_model,  X_test_scaled, y_test, "Decision Tree")

    X_test_lead, _ = get_features(test_df, feature_mode="lead_only")
    if X_test_lead.isnull().any().any():
        X_test_lead = pd.DataFrame(
            lead_imputer.transform(X_test_lead), columns=lead_cols)
    dt_opt_results = evaluate(dt_opt_model, X_test_lead, y_test, "DT Optimized")

    # Model comparison table (all models × both feature modes)
    model_comparison = compare_models(
        train_df, val_df, test_df, y_train, y_val, y_test,
        dt_opt_model=dt_opt_model,
        dt_opt_threshold=dt_opt_threshold,
        dt_opt_feature_cols=lead_cols,
    )

    # Compute SHAP importance for Random Forest
    sample_size = min(100, len(X_test_scaled))
    X_sample = X_test_scaled.sample(n=sample_size, random_state=RANDOM_STATE)
    shap_importance_dict = shap_importance(rf_model, X_sample, X_test_scaled.columns.tolist())

    class_prevalence = y_test.mean()
    dt_rules = export_text(dt_model, feature_names=X_test_scaled.columns.tolist())

    metadata = {
        'feature_columns':           X_test_scaled.columns.tolist(),
        'lead_only_feature_columns': lead_cols,
        'trained_at':                pd.Timestamp.now().isoformat(),
        'class_prevalence':          class_prevalence,
        'n_samples':                 len(df),
        'n_features':                len(X_test_scaled.columns),
        'dt_rules':                  dt_rules,
        'dt_optimized_rules':        dt_opt_rules,
        'dt_optimized_threshold':    dt_opt_threshold,
        'rf_gini_importance':        dict(zip(X_test_scaled.columns, rf_model.feature_importances_)),
        'model_comparison':          model_comparison,
    }

    joblib.dump(metadata, MODELS_DIR / 'classification_metadata.joblib')

    result = {
        'random_forest':       rf_results,
        'xgboost':             xgb_results,
        'decision_tree':       dt_results,
        'decision_tree_optimized': dt_opt_results,
        'shap_importance':     shap_importance_dict,
        'metadata':            metadata,
    }

    logger.info("Classification pipeline complete")
    return result

if __name__ == "__main__":
    # Test the classification engine
    logger.info("Testing classification engine...")

    # Load the processed panel
    df = pd.read_parquet('data/processed/panel_pca.parquet')

    # Run classification
    result = run_all(df)

    rf_auc = result['random_forest'].get('roc_auc', 0)
    rf_rec = result['random_forest'].get('recall', 0)

    print(f'RF ROC-AUC: {rf_auc}')
    print(f'RF Recall: {rf_rec}')
    print(f'XGB ROC-AUC: {result["xgboost"].get("roc_auc", 0)}')
    print(f'False Negatives: {result["random_forest"]["false_negatives"]}')

    assert rf_auc > 0.5, f'ROC-AUC too low: {rf_auc}'
    print('CLASSIFICATION TEST: PASSED')