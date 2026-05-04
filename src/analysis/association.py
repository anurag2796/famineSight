# src/analysis/association.py
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import Dict, Any, List
import logging
from pathlib import Path
from src.config import FP_MIN_SUPPORT, APRIORI_MIN_CONFIDENCE, APRIORI_MIN_LIFT, MODELS_DIR, DATA_PROC
from src.data.preprocessor import load_and_merge
from prefixspan import PrefixSpan
import json

logger = logging.getLogger(__name__)

def binarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create boolean transaction DataFrame from the panel data.

    Thresholds use the actual data distribution:
    - IPC columns are 0-1 fractions (e.g. 0.10 = 10% of population)
    - Food price index is normalized to historical median=100; Somalia's
      chronic price crisis means panel values are all ~220-346, so
      spike/moderate are defined relative to the panel's own quartiles.
    """
    p75_price = df['food_price_index'].quantile(0.75)
    p50_price = df['food_price_index'].quantile(0.50)

    transaction_df = pd.DataFrame({
        'drought_severe':    (df['rainfall_anomaly_pct'] < -30),
        'drought_moderate':  (df['rainfall_anomaly_pct'].between(-30, -15, inclusive='left')),
        'conflict_high':     (df['conflict_fatalities'] >= 10),
        'conflict_low':      (df['conflict_fatalities'].between(1, 10, inclusive='left')),
        # price thresholds relative to panel distribution
        'price_spike':       (df['food_price_index'] > p75_price),
        'price_moderate':    (df['food_price_index'].between(p50_price, p75_price, inclusive='left')),
        # IPC columns are fractions (0-1); phase5 is clipped to 0 by preprocessor
        'ipc_phase4_high':   (df['ipc_phase4_pct'] >= 0.10),
        'ipc_phase3_high':   (df['ipc_phase3_pct'] >= 0.30),
        'crisis':            (df['crisis_label'] == 1),
    })

    return transaction_df

def run_fpgrowth(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run FP-Growth algorithm on transaction data.

    Args:
        trans_df: Boolean transaction DataFrame

    Returns:
        DataFrame with FP-Growth rules sorted by lift
    """
    logger.info("Running FP-Growth...")

    # Run FP-Growth
    frequent_itemsets = fpgrowth(trans_df, min_support=FP_MIN_SUPPORT, use_colnames=True)

    if frequent_itemsets.empty:
        logger.warning("No frequent itemsets found with FP-Growth")
        return pd.DataFrame()

    # Generate association rules
    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=APRIORI_MIN_LIFT
    )

    # Filter by confidence
    rules = rules[rules['confidence'] >= APRIORI_MIN_CONFIDENCE]

    # Sort by lift (descending)
    rules = rules.sort_values('lift', ascending=False)

    logger.info(f"FP-Growth found {len(rules)} rules")

    return rules

def run_apriori(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Apriori algorithm on transaction data.

    Args:
        trans_df: Boolean transaction DataFrame

    Returns:
        DataFrame with Apriori rules sorted by lift
    """
    logger.info("Running Apriori...")

    # Run Apriori
    frequent_itemsets = apriori(trans_df, min_support=FP_MIN_SUPPORT, use_colnames=True)

    if frequent_itemsets.empty:
        logger.warning("No frequent itemsets found with Apriori")
        return pd.DataFrame()

    # Generate association rules
    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=APRIORI_MIN_LIFT
    )

    # Filter by confidence
    rules = rules[rules['confidence'] >= APRIORI_MIN_CONFIDENCE]

    # Sort by lift (descending)
    rules = rules.sort_values('lift', ascending=False)

    logger.info(f"Apriori found {len(rules)} rules")

    return rules

def build_sequences(df: pd.DataFrame) -> List[List[str]]:
    """
    Convert district time series to sequences of events for PrefixSpan.
    Price thresholds match binarize() — relative to panel distribution.
    """
    logger.info("Building sequences for PrefixSpan...")

    p75_price = df['food_price_index'].quantile(0.75)
    p50_price = df['food_price_index'].quantile(0.50)

    sequences = []
    for pcode, group in df.groupby('pcode'):
        group = group.sort_values('date')
        sequence = []
        for _, row in group.iterrows():
            event = ""
            if row['rainfall_anomaly_pct'] < -30:
                event += "D"
            elif -30 <= row['rainfall_anomaly_pct'] < -15:
                event += "d"
            if row['conflict_fatalities'] >= 10:
                event += "C"
            elif 1 <= row['conflict_fatalities'] < 10:
                event += "c"
            if row['food_price_index'] > p75_price:
                event += "P"
            elif p50_price <= row['food_price_index'] <= p75_price:
                event += "p"
            if row['crisis_label'] == 1:
                event += "M"
            if event:
                sequence.append(event)
        if sequence:
            sequences.append(sequence)

    logger.info(f"Built {len(sequences)} sequences")
    return sequences

def run_sequential(sequences: List[List[str]]) -> pd.DataFrame:
    """
    Run sequential pattern mining using PrefixSpan.

    Args:
        sequences: List of sequences

    Returns:
        DataFrame with sequential patterns
    """
    logger.info("Running PrefixSpan sequential pattern mining...")

    if not sequences:
        logger.warning("No sequences provided")
        return pd.DataFrame()

    # Minimum support (e.g., 20% of districts must have this pattern)
    min_sup = max(2, int(len(sequences) * 0.2))

    ps = PrefixSpan(sequences)
    frequent_patterns = ps.frequent(min_sup)
    
    # Sort by frequency (descending)
    frequent_patterns.sort(key=lambda x: x[0], reverse=True)

    # Format into DataFrame
    patterns_data = []
    for freq, pattern in frequent_patterns:
        patterns_data.append({
            'pattern': ' -> '.join(pattern),
            'frequency': freq,
            'support': freq / len(sequences)
        })

    patterns_df = pd.DataFrame(patterns_data)
    
    # Filter for patterns of length >= 2
    if not patterns_df.empty:
        patterns_df = patterns_df[patterns_df['pattern'].str.contains(' -> ')]
        
    logger.info(f"PrefixSpan found {len(patterns_df)} sequential patterns of length >= 2")
    return patterns_df

def run_all(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all association rule mining methods.

    Args:
        df: Master panel DataFrame

    Returns:
        Dictionary with results from all methods
    """
    logger.info("Starting association rule mining...")

    # Create transaction data
    transaction_df = binarize(df)

    # Save transaction data
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    transaction_df.to_parquet(DATA_PROC / 'transactions.parquet')

    # Run FP-Growth
    fp_rules = run_fpgrowth(transaction_df)

    # Run Apriori
    apriori_rules = run_apriori(transaction_df)

    # Run sequential pattern mining
    sequences = build_sequences(df)
    sequential_patterns = run_sequential(sequences)

    # Calculate statistics
    stats = {
        'fp_n_rules': len(fp_rules),
        'apriori_n_rules': len(apriori_rules),
        'seq_n_patterns': len(sequential_patterns),
        'crisis_prevalence': df['crisis_label'].mean()
    }

    # Return results
    result = {
        'fpgrowth_rules': fp_rules,
        'apriori_rules': apriori_rules,
        'sequential_patterns': sequential_patterns,
        'stats': stats
    }
    
    # Prepare serializable result for JSON
    serializable_result = {
        'fpgrowth_rules': fp_rules.to_dict(orient='records') if not fp_rules.empty else [],
        'apriori_rules': apriori_rules.to_dict(orient='records') if not apriori_rules.empty else [],
        'sequential_patterns': sequential_patterns.to_dict(orient='records') if not sequential_patterns.empty else [],
        'stats': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in stats.items()}
    }
    
    # Save results to MODELS_DIR (where model_registry loads from)
    output_path = MODELS_DIR / 'association_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Need to handle frozensets in the rules dataframe dictionaries
    def set_default(obj):
        if isinstance(obj, frozenset):
            return list(obj)
        raise TypeError
        
    with open(output_path, 'w') as f:
        json.dump(serializable_result, f, default=set_default, indent=2)

    logger.info(f"Association rule mining complete. Saved to {output_path}")
    return result

if __name__ == "__main__":
    # Test the association analysis
    logger.info("Testing association analysis...")

    # Load the master panel
    df = pd.read_parquet('data/processed/master_panel.parquet')

    # Run all association analysis
    result = run_all(df)

    logger.info(f"Results: {result['stats']}")
    print("Association analysis complete!")