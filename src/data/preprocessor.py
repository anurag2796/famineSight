# src/data/preprocessor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.config import (
    DATA_RAW, DATA_PROC, ALL_FEATURES, TARGET_COL,
    LAG_MONTHS, RF_N_JOBS, RANDOM_STATE, MODELS_DIR
)
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_merge():
    """
    Load all datasets and merge them into a single panel
    """
    logger.info("Loading and merging datasets...")

    # Load datasets
    try:
        # Load CHIRPS (rainfall) data
        chirps_path = DATA_RAW / "chirps" / "chirps_rainfall.csv"
        if chirps_path.exists():
            chirps_df = pd.read_csv(chirps_path)
            chirps_df['date'] = pd.to_datetime(chirps_df['date'])
            logger.info(f"Loaded CHIRPS data: {chirps_df.shape}")
        else:
            logger.warning("CHIRPS data not found")
            chirps_df = pd.DataFrame()

        # Load ACLED data
        acled_path = DATA_RAW / "acled" / "somalia_acled.csv"
        if acled_path.exists():
            acled_df = pd.read_csv(acled_path)
            # ACLED now has 'date' column as first of month, but be robust
            for date_col in ['date', 'event_date', 'event_month']:
                if date_col in acled_df.columns:
                    acled_df['date'] = pd.to_datetime(acled_df[date_col])
                    break
            
            if 'date' not in acled_df.columns:
                logger.error("ACLED data missing date column")
                acled_df = pd.DataFrame()
            else:
                logger.info(f"Loaded ACLED data: {acled_df.shape}")
        else:
            logger.warning("ACLED data not found")
            acled_df = pd.DataFrame()

        # Load WFP data
        wfp_path = DATA_RAW / "wfp" / "wfp_prices_som.csv"
        if wfp_path.exists():
            wfp_df = pd.read_csv(wfp_path)
            wfp_df['date'] = pd.to_datetime(wfp_df['date'])
            logger.info(f"Loaded WFP data: {wfp_df.shape}")
        else:
            logger.warning("WFP data not found")
            wfp_df = pd.DataFrame()

        # Load FSNAU data
        fsnau_path = DATA_RAW / "fsnau" / "fsnau_mortality.csv"
        if fsnau_path.exists():
            fsnau_df = pd.read_csv(fsnau_path)
            fsnau_df['date'] = pd.to_datetime(fsnau_df['date'])
            logger.info(f"Loaded FSNAU data: {fsnau_df.shape}")
        else:
            logger.warning("FSNAU data not found")
            fsnau_df = pd.DataFrame()

        # Load IPC data
        ipc_path = DATA_RAW / "ipc" / "ipc_phases.csv"
        if ipc_path.exists():
            ipc_df = pd.read_csv(ipc_path)
            ipc_df['date'] = pd.to_datetime(ipc_df['date'])
            logger.info(f"Loaded IPC data: {ipc_df.shape}")
        else:
            logger.warning("IPC data not found")
            ipc_df = pd.DataFrame()

        # Load NDVI data
        ndvi_path = DATA_RAW / "ndvi" / "ndvi_somalia.csv"
        if ndvi_path.exists():
            ndvi_df = pd.read_csv(ndvi_path)
            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
            logger.info(f"Loaded NDVI data: {ndvi_df.shape}")
        else:
            logger.warning("NDVI data not found — ndvi_anomaly will be NaN (imputed later)")
            ndvi_df = pd.DataFrame()

        # Load UNHCR displacement data
        unhcr_path = DATA_RAW / "unhcr" / "unhcr_displacement_som.csv"
        if unhcr_path.exists():
            unhcr_df = pd.read_csv(unhcr_path)
            unhcr_df['date'] = pd.to_datetime(unhcr_df['date'])
            logger.info(f"Loaded UNHCR displacement data: {unhcr_df.shape}")
        else:
            logger.warning("UNHCR data not found — displacement features will be NaN (imputed later)")
            unhcr_df = pd.DataFrame()

        # ------------------------------------------------------------------
        # Merge strategy:
        #   BASE  : IPC (real data, 2017-2026, 15 pcodes x ~22 time points)
        #   crisis_label derived from IPC phase4+5 >= 10% threshold
        #   FSNAU : supplementary mortality rates (left-join, sparse OK)
        #   Others: left-join on (date, pcode)
        # ------------------------------------------------------------------
        logger.info("Merging datasets...")

        if ipc_df.empty:
            logger.error("IPC data is empty — cannot build master panel without it.")
            raise ValueError("IPC data required as merge base but is empty.")

        # ---- IPC as base: keep all IPC phase columns -----------------------
        ipc_cols = ['date', 'pcode', 'ipc_phase1_pct', 'ipc_phase2_pct',
                    'ipc_phase3_pct', 'ipc_phase4_pct', 'ipc_phase5_pct']
        ipc_cols_available = [c for c in ipc_cols if c in ipc_df.columns]
        master_df = ipc_df[ipc_cols_available].copy()

        # ---- Derive crisis_label from IPC phase 4 + phase 5 ---------------
        p4 = master_df.get('ipc_phase4_pct', pd.Series(0.0, index=master_df.index))
        p5 = master_df.get('ipc_phase5_pct', pd.Series(0.0, index=master_df.index))
        master_df[TARGET_COL] = ((p4.fillna(0) + p5.fillna(0)) >= 0.10).astype(int)
        logger.info(
            f"Derived crisis_label from IPC (phase4+phase5 >= 10%): "
            f"{master_df[TARGET_COL].sum()} positive / {len(master_df)} total "
            f"({master_df[TARGET_COL].mean():.1%})"
        )

        # ---- CHIRPS rainfall -----------------------------------------------
        if not chirps_df.empty:
            master_df = master_df.merge(
                chirps_df[['date', 'pcode', 'rainfall_anomaly_pct']],
                on=['date', 'pcode'], how='left'
            )
            logger.info("Merged CHIRPS data")

        # ---- WFP food prices -----------------------------------------------
        if not wfp_df.empty:
            master_df = master_df.merge(
                wfp_df[['date', 'pcode', 'food_price_index']],
                on=['date', 'pcode'], how='left'
            )
            logger.info("Merged WFP data")

        # ---- ACLED conflict -------------------------------------------------
        if not acled_df.empty:
            acled_agg = acled_df.groupby(['date', 'pcode']).agg({
                'conflict_events': 'sum',
                'conflict_fatalities': 'sum',
                'civilian_targeting_events': 'sum'
            }).reset_index()
            master_df = master_df.merge(
                acled_agg, on=['date', 'pcode'], how='left'
            )
            logger.info("Merged ACLED data")

        # ---- FSNAU mortality (supplementary) --------------------------------
        if not fsnau_df.empty:
            fsnau_agg = fsnau_df.groupby(['date', 'pcode']).agg({
                'cdr_per_10k_per_day': 'mean',
                'u5dr_per_10k_per_day': 'mean'
            }).reset_index()
            master_df = master_df.merge(
                fsnau_agg[['date', 'pcode', 'cdr_per_10k_per_day', 'u5dr_per_10k_per_day']],
                on=['date', 'pcode'], how='left'
            )
            logger.info("Merged FSNAU mortality data (supplementary)")
        else:
            master_df['cdr_per_10k_per_day'] = np.nan
            master_df['u5dr_per_10k_per_day'] = np.nan

        # ---- NDVI vegetation anomaly ----------------------------------------
        if not ndvi_df.empty:
            master_df = master_df.merge(
                ndvi_df[['date', 'pcode', 'ndvi_anomaly']],
                on=['date', 'pcode'], how='left'
            )
            logger.info(
                f"Merged NDVI data: "
                f"{master_df['ndvi_anomaly'].notna().sum()} non-null values"
            )
        else:
            master_df['ndvi_anomaly'] = np.nan

        # ---- UNHCR displacement (IDPs + refugees) ---------------------------
        if not unhcr_df.empty:
            unhcr_agg = unhcr_df.groupby(['date', 'pcode']).agg({
                'idp_count': 'sum',
                'refugee_count': 'sum'
            }).reset_index()
            master_df = master_df.merge(
                unhcr_agg, on=['date', 'pcode'], how='left'
            )
            logger.info(
                f"Merged UNHCR displacement data: "
                f"{master_df['idp_count'].notna().sum()} non-null IDP values"
            )
        else:
            master_df['idp_count'] = np.nan
            master_df['refugee_count'] = np.nan

        logger.info(f"Master panel shape after merge: {master_df.shape}")

        # ---- Validation ----------------------------------------------------
        crisis_rate = master_df[TARGET_COL].mean()
        logger.info(f"Crisis rate after merge: {crisis_rate:.1%}")
        if crisis_rate == 0.0:
            logger.warning("Crisis rate is 0% — model will have no positive labels to learn from!")
        elif crisis_rate > 0.5:
            logger.warning(f"Crisis rate is very high ({crisis_rate:.1%}) — check IPC threshold.")
        if len(master_df) < 100:
            logger.warning(f"Panel has only {len(master_df)} rows — may be too small for training.")

        return master_df

    except Exception as e:
        logger.error(f"Error in load_and_merge: {e}")
        raise

def temporal_sort(df):
    """
    Sort data chronologically by pcode and date
    """
    logger.info("Sorting data temporally...")

    if df.empty:
        logger.warning("Empty DataFrame, skipping temporal sort")
        return df

    try:
        # Sort by pcode and date
        df_sorted = df.sort_values(['pcode', 'date'])
        logger.info("Temporal sorting complete")
        return df_sorted
    except Exception as e:
        logger.error(f"Error in temporal_sort: {e}")
        raise

def impute_missing(df):
    """
    Impute missing values using per-district median and KNN imputation
    """
    logger.info("Imputing missing values...")

    if df.empty:
        logger.warning("Empty DataFrame, skipping imputation")
        return df

    try:
        from src.config import TARGET_COL, AUX_TARGETS
        
        # Log missing counts before imputation
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before imputation: {missing_before}")

        # Stage 0: Drop rows where target is missing (we can't train/eval on these)
        df_filled = df.dropna(subset=[TARGET_COL]).copy()

        # Stage 0.5: Fill AUX_TARGET columns (FSNAU mortality \u2014 very sparse)
        # with global median so they don\u2019t pollute KNN imputation.
        for col in AUX_TARGETS:
            if col in df_filled.columns:
                col_median = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(col_median if not np.isnan(col_median) else 0.0)

        # Stage 1: Per-district median fill for MCAR
        # Skip targets and identifiers
        skip_cols = ['date', 'district', 'pcode', TARGET_COL] + AUX_TARGETS
        
        for col in df_filled.columns:
            if col in skip_cols:
                continue
            if df_filled[col].isnull().any():
                # Fill with median per district
                medians = df_filled.groupby('pcode')[col].transform('median')
                df_filled[col] = df_filled[col].fillna(medians)
                # If still NaN (entire district has no values), fill with global median
                global_med = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(global_med if not np.isnan(global_med) else 0.0)

        # Stage 2: KNN imputation for remaining NaN values
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in skip_cols]

        if len(numeric_cols) > 0:
            # Check if we have remaining NaNs in numeric features
            if df_filled[numeric_cols].isnull().sum().sum() > 0:
                imputer = KNNImputer(n_neighbors=5, weights='distance', keep_empty_features=True)
                df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])

        # Fill any remaining NaNs with global median as last resort
        for col in numeric_cols:
            if df_filled[col].isnull().any():
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())

        missing_after = df_filled.isnull().sum().sum()
        logger.info(f"Missing values after imputation: {missing_after}")

        return df_filled

    except Exception as e:
        logger.error(f"Error in impute_missing: {e}")
        raise

def clip_outliers(df):
    """
    Clip outliers using IQR method
    """
    logger.info("Clipping outliers...")

    if df.empty:
        logger.warning("Empty DataFrame, skipping outlier clipping")
        return df

    try:
        # Identify numeric columns
        from src.config import TARGET_COL, AUX_TARGETS
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['pcode', 'date', TARGET_COL] + AUX_TARGETS]

        # Create a copy to work on
        df_clipped = df.copy()

        # For each numeric column, clip using IQR
        for col in numeric_cols:
            if df_clipped[col].isnull().any():
                continue # Skip if NaN (though they should be imputed by now)
            
            Q1 = df_clipped[col].quantile(0.25)
            Q3 = df_clipped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            # Clip values
            df_clipped[col] = df_clipped[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info("Outlier clipping complete")
        return df_clipped

    except Exception as e:
        logger.error(f"Error in clip_outliers: {e}")
        raise

def engineer_lag_features(df):
    """
    Engineer lag features for climate, conflict, and market features
    """
    logger.info("Engineering lag features...")

    if df.empty:
        logger.warning("Empty DataFrame, skipping lag feature engineering")
        return df, []

    try:
        # Get feature columns to lag
        feature_cols = [col for col in df.columns if col in ALL_FEATURES]

        # Create lag features for each feature
        lagged_features = []

        for col in feature_cols:
            for lag in LAG_MONTHS:
                lag_col_name = f"{col}_lag_{lag}m"
                df[lag_col_name] = df.groupby('pcode')[col].shift(lag)
                lagged_features.append(lag_col_name)

        # Also create rolling averages for certain features
        rolling_features = ['rainfall_anomaly_pct', 'conflict_fatalities', 'food_price_index']
        for col in rolling_features:
            if col in df.columns:
                for window in [3]:  # 3-month rolling average
                    rolling_col_name = f"{col}_rolling_{window}m"
                    df[rolling_col_name] = df.groupby('pcode')[col].rolling(window=window).mean().reset_index(0, drop=True)
                    lagged_features.append(rolling_col_name)

        # Drop rows with NaN values introduced by lagging
        df = df.dropna(subset=lagged_features)

        logger.info(f"Lag feature engineering complete. Generated {len(lagged_features)} lag features")
        return df, lagged_features

    except Exception as e:
        logger.error(f"Error in engineer_lag_features: {e}")
        raise

def scale_features(df, fit=True):
    """
    Scale features to [0, 1] range using MinMaxScaler
    """
    logger.info("Scaling features...")

    if df.empty:
        logger.warning("Empty DataFrame, skipping feature scaling")
        return df

    try:
        # Identify numeric columns (excluding date, pcode, district, target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['pcode', 'date', 'district', TARGET_COL]]

        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for scaling")
            return df

        # Create scaler
        scaler = MinMaxScaler()

        if fit:
            # Fit and transform
            df_scaled = df.copy()
            df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            # Save scaler
            scaler_path = MODELS_DIR / "scaler.joblib"
            import joblib
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        else:
            # Load scaler and transform
            scaler_path = MODELS_DIR / "scaler.joblib"
            if scaler_path.exists():
                import joblib
                scaler = joblib.load(scaler_path)
                df_scaled = df.copy()
                df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
                logger.info("Loaded scaler and applied to data")
            else:
                logger.warning("Scaler not found, skipping scaling")
                return df

        logger.info("Feature scaling complete")
        return df_scaled

    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        raise

def apply_pca(df, fit=True):
    """
    Apply PCA to environmental and market features
    """
    logger.info("Applying PCA...")

    if df.empty:
        logger.warning("Empty DataFrame, skipping PCA")
        return df, None

    try:
        # Identify features to apply PCA to (environmental + market features)
        pca_features = [col for col in df.columns if col in ALL_FEATURES]

        # Remove any non-numeric columns
        numeric_pca_features = [col for col in pca_features if df[col].dtype in ['int64', 'float64']]

        if len(numeric_pca_features) == 0:
            logger.warning("No numeric features found for PCA")
            return df, None

        # Create PCA
        pca = PCA(n_components=0.95)  # Retain 95% variance

        if fit:
            # Fit and transform
            df_pca = df.copy()
            df_transformed = pca.fit_transform(df[numeric_pca_features])
            df_pca = df.copy()

            # Create new column names for PCA components
            pca_col_names = [f"pca_comp_{i}" for i in range(df_transformed.shape[1])]

            # Add PCA components to DataFrame
            for i, col_name in enumerate(pca_col_names):
                df_pca[col_name] = df_transformed[:, i]

            # Save PCA model
            pca_path = MODELS_DIR / "pca.joblib"
            import joblib
            joblib.dump(pca, pca_path)
            logger.info(f"Saved PCA model to {pca_path}")
            logger.info(f"PCA reduced features from {len(numeric_pca_features)} to {len(pca_col_names)} components")
        else:
            # Load PCA and transform
            pca_path = MODELS_DIR / "pca.joblib"
            if pca_path.exists():
                import joblib
                pca = joblib.load(pca_path)
                df_transformed = pca.transform(df[numeric_pca_features])
                df_pca = df.copy()

                # Create new column names for PCA components
                pca_col_names = [f"pca_comp_{i}" for i in range(df_transformed.shape[1])]

                # Add PCA components to DataFrame
                for i, col_name in enumerate(pca_col_names):
                    df_pca[col_name] = df_transformed[:, i]

                logger.info("Loaded PCA model and applied to data")
            else:
                logger.warning("PCA model not found, skipping PCA")
                return df, None

        logger.info("PCA application complete")
        return df_pca, pca

    except Exception as e:
        logger.error(f"Error in apply_pca: {e}")
        raise

def run_full_pipeline(save=True):
    """
    Run the complete preprocessing pipeline
    """
    logger.info("Starting full preprocessing pipeline...")

    # Load and merge data
    df = load_and_merge()

    # Temporal sort
    df = temporal_sort(df)

    # Impute missing values
    df = impute_missing(df)

    # Clip outliers
    df = clip_outliers(df)

    # Engineer lag features
    df, lag_features = engineer_lag_features(df)

    if save:
        # Save master panel BEFORE scaling/PCA (contains raw features with lag engineering)
        output_path = DATA_PROC / "master_panel.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved master panel to {output_path}")

    # Scale features
    df_scaled = scale_features(df, fit=True)

    if save:
        # Save scaled panel (MinMax normalized, no PCA)
        output_path_scaled = DATA_PROC / "panel_scaled.parquet"
        df_scaled.to_parquet(output_path_scaled, index=False)
        logger.info(f"Saved scaled panel to {output_path_scaled}")

    # Apply PCA
    df_pca, pca_model = apply_pca(df_scaled, fit=True)

    if save:
        # Save PCA panel (scaled features + PCA components)
        output_path_pca = DATA_PROC / "panel_pca.parquet"
        df_pca.to_parquet(output_path_pca, index=False)
        logger.info(f"Saved PCA panel to {output_path_pca}")

    logger.info("Preprocessing pipeline complete")
    return df_pca

# Test function
def test_preprocessor():
    """
    Test the preprocessor with synthetic data
    """
    logger.info("Testing preprocessor with synthetic data...")

    try:
        # Run full pipeline
        df = run_full_pipeline(save=True)

        logger.info(f"Panel shape: {df.shape}")
        logger.info(f"Crisis rate: {df[TARGET_COL].mean() if TARGET_COL in df.columns else 'N/A'}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")

        assert df.isnull().sum().sum() == 0, "Still has missing values!"
        logger.info("PREPROCESSOR TEST: PASSED")

        return True
    except Exception as e:
        logger.error(f"Preprocessor test failed: {e}")
        return False

if __name__ == "__main__":
    test_preprocessor()