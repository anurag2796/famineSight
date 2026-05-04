import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
from pathlib import Path
import joblib

ROOT = Path("/home/anurag/codebase/famineSight")
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate Plots
def generate_plots():
    print("Generating plots...")
    
    viz_file = MODELS_DIR / "viz_payload.json"
    if not viz_file.exists():
        print("viz_payload.json not found. Exiting plot generation.")
        return
        
    with open(viz_file, 'r') as f:
        viz = json.load(f)

    # Load data
    master_file = DATA_DIR / "master_panel.parquet"
    if master_file.exists():
        df = pd.read_parquet(master_file)
        
        # 1. Crisis events over time
        plt.figure(figsize=(10, 5))
        ts = df.groupby('date')['crisis_label'].sum().reset_index()
        plt.plot(ts['date'], ts['crisis_label'], color='red', lw=2)
        plt.fill_between(ts['date'], ts['crisis_label'], color='red', alpha=0.3)
        plt.title('Total District-Months in Crisis over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Crises')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "crisis_trend.png", dpi=300)
        plt.close()

        # 2. Correlation Heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        focus_cols = ['rainfall_anomaly_1mo_lag', 'price_index', 'conflict_fatalities', 'fatalities_rolling_3mo', 'ndvi_anomaly', 'crisis_label']
        focus_cols = [c for c in focus_cols if c in numeric_cols]
        if len(focus_cols) > 2:
            plt.figure(figsize=(8, 6))
            sns.heatmap(df[focus_cols].corr(), annot=True, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.title('Correlation Matrix of Key Indicators')
            plt.tight_layout()
            plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=300)
            plt.close()

        # 3. Distributions Plot
        if 'price_index' in df.columns and 'crisis_label' in df.columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x='crisis_label', y='price_index', palette='Set2')
            plt.title('Food Price Index Distribution by Crisis State')
            plt.xlabel('Crisis State (0 = Stable, 1 = Crisis)')
            plt.ylabel('Normalized Price Index')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "price_distribution.png", dpi=300)
            plt.close()

        # 4. Outlier PCT
        df_numeric = df.select_dtypes(include=[np.number])
        z_scores = np.abs((df_numeric - df_numeric.mean()) / df_numeric.std())
        outliers = (z_scores > 3).mean() * 100
        plt.figure(figsize=(10, 5))
        outliers.sort_values(ascending=False).head(15).plot(kind='bar', color='skyblue')
        plt.title('Top 15 Features by Outlier Percentage (Z-Score > 3)')
        plt.ylabel('Outlier Percentage (%)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "outlier_pct.png", dpi=300)
        plt.close()

    # 5. Missing PCT
    missing_data = viz.get("feature_engineering", {}).get("missing_pct", {})
    if missing_data:
        plt.figure(figsize=(10, 5))
        series = pd.Series(missing_data).sort_values(ascending=False).head(15)
        series.plot(kind='bar', color='coral')
        plt.title('Top 15 Features by Missing Percentage (Before Imputation)')
        plt.ylabel('Missing Percentage (%)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "missing_pct.png", dpi=300)
        plt.close()

    # 6. Feature Selection (Chi2 & MI)
    fs = viz.get("feature_selection")
    if fs and fs.get("features"):
        df_fs = pd.DataFrame({"feature": fs["features"], "chi2": fs["chi2"], "mi": fs["mutual_info"]})
        df_fs = df_fs.sort_values("mi", ascending=True).tail(15)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].barh(df_fs["feature"], df_fs["chi2"], color="#f87171")
        axes[0].set_title("Chi2 Score")
        axes[1].barh(df_fs["feature"], df_fs["mi"], color="#34d399")
        axes[1].set_title("Mutual Information")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "feature_selection.png", dpi=300)
        plt.close()

    # 7. Coverage Matrix
    dq = viz.get("data_quality", {})
    cov = dq.get("coverage")
    if cov:
        z = np.array(cov["matrix"])
        plt.figure(figsize=(12, 6))
        sns.heatmap(z.T, xticklabels=cov["months"], yticklabels=cov["features"], cmap="viridis", cbar_kws={'label': 'Coverage'})
        # Thin out x-ticks so it doesn't overlap
        n = len(cov["months"])
        step = max(1, n // 10)
        plt.xticks(np.arange(0, n, step), [cov["months"][i] for i in range(0, n, step)], rotation=45)
        plt.title("Month × Feature Data Availability Coverage")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "coverage_matrix.png", dpi=300)
        plt.close()

    # 8. Lag Autocorrelation
    fe = viz.get("feature_engineering", {})
    acf = fe.get("lag_autocorr")
    if acf:
        plt.figure(figsize=(8, 6))
        sns.heatmap(acf["matrix"], xticklabels=[f"lag {l}" for l in acf["lags"]], yticklabels=acf["features"], cmap="RdBu", center=0, cbar_kws={'label': 'Autocorrelation (ρ)'})
        plt.title("Lag Autocorrelation per Feature")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "lag_autocorr.png", dpi=300)
        plt.close()

    # 9. ROC and PR Curves
    curves = viz.get("classification_curves", {}).get("models", {})
    if curves:
        plt.figure(figsize=(8, 6))
        for name, d in curves.items():
            roc = d.get("roc")
            if roc:
                plt.plot(roc["fpr"], roc["tpr"], lw=2, label=f"{name} (AUC={d.get('roc_auc', 0):.2f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "roc_curves.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        for name, d in curves.items():
            pr = d.get("pr")
            if pr:
                plt.plot(pr["recall"], pr["precision"], lw=2, label=f"{name} (AP={d.get('avg_precision', 0):.2f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "pr_curves.png", dpi=300)
        plt.close()
        
        # 10. Calibration and Confusion Matrix (Random Forest specifically)
        rf_curve = curves.get("Random Forest")
        if rf_curve:
            # Calibration
            cal = rf_curve.get("calibration")
            if cal:
                plt.figure(figsize=(6, 5))
                plt.plot(cal["prob_pred"], cal["prob_true"], marker='o', lw=2, label="Random Forest")
                plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Perfectly calibrated")
                plt.xlabel("Mean predicted probability")
                plt.ylabel("Fraction of positives")
                plt.title("Calibration Curve (Random Forest)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(FIG_DIR / "calibration_curve.png", dpi=300)
                plt.close()
            # Confusion Matrix
            cm = rf_curve.get("confusion_matrix")
            if cm:
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
                plt.title("Confusion Matrix (Random Forest)")
                plt.tight_layout()
                plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=300)
                plt.close()

    # 11. PCA Variance
    pca_data = viz.get("pca_diagnostics")
    if pca_data:
        plt.figure(figsize=(8, 5))
        x = range(1, pca_data["n_components"] + 1)
        plt.bar(x, pca_data["explained_variance"], alpha=0.7, label='Explained Variance')
        plt.plot(x, pca_data["cumulative"], color='red', marker='o', label='Cumulative Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('PCA Scree Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "pca_scree.png", dpi=300)
        plt.close()

    # 12. Clustering Elbow
    elbow = viz.get("clustering_extras", {}).get("elbow")
    if elbow:
        df_elbow = pd.DataFrame(elbow)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = 'tab:blue'
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Sum of Squared Errors (SSE)', color=color)
        ax1.plot(df_elbow["k"], df_elbow["sse"], marker='o', color=color, lw=2, label="SSE")
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Silhouette Score', color=color2)
        ax2.plot(df_elbow["k"], df_elbow["silhouette"], marker='s', color=color2, lw=2, label="Silhouette")
        ax2.tick_params(axis='y', labelcolor=color2)
        plt.title('K-Means Elbow Method & Silhouette')
        fig.tight_layout()
        plt.savefig(FIG_DIR / "clustering_elbow.png", dpi=300)
        plt.close()

    # 13. Association Rules
    assoc_file = MODELS_DIR / "association_results.json"
    if assoc_file.exists():
        with open(assoc_file, 'r') as f:
            assoc = json.load(f)
        rules = assoc.get("fpgrowth_rules", [])
        if rules:
            df_rules = pd.DataFrame(rules)
            if 'confidence' in df_rules.columns and 'lift' in df_rules.columns:
                plt.figure(figsize=(8, 6))
                support_col = 'support' if 'support' in df_rules.columns else 'confidence'
                scatter = plt.scatter(df_rules['confidence'], df_rules['lift'], c=df_rules[support_col], cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Support')
                plt.xlabel('Confidence')
                plt.ylabel('Lift')
                plt.title('Association Rules Scatter Plot (FP-Growth)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(FIG_DIR / "association_rules.png", dpi=300)
                plt.close()

    # 14. t-SNE Plot
    tsne_data = viz.get("eda", {}).get("tsne", [])
    if tsne_data:
        df_tsne = pd.DataFrame(tsne_data)
        plt.figure(figsize=(8, 6))
        if 'label' in df_tsne.columns:
            sns.scatterplot(data=df_tsne, x='x', y='y', hue='label', palette='coolwarm', alpha=0.7)
        else:
            plt.scatter(df_tsne['x'], df_tsne['y'], alpha=0.7)
        plt.title('t-SNE Projection of District Profiles')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "tsne_plot.png", dpi=300)
        plt.close()
            
    # 15. Anomaly Scores
    anomaly_file = MODELS_DIR / "anomaly_results.json"
    if anomaly_file.exists():
        with open(anomaly_file, 'r') as f:
            anomaly = json.load(f)
        scores = anomaly.get("isolation_forest", {}).get("scores", [])
        if scores:
            plt.figure(figsize=(8, 5))
            plt.hist(scores, bins=50, color='purple', alpha=0.7)
            plt.title('Distribution of Isolation Forest Anomaly Scores')
            plt.xlabel('Anomaly Score (Lower is more anomalous)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "anomaly_scores.png", dpi=300)
            plt.close()
            
    # 16. Algorithm Zoo Comparison
    extra_models_file = MODELS_DIR / "classification_metadata.joblib"
    if extra_models_file.exists():
        try:
            meta = joblib.load(extra_models_file)
            
            # Extract Gini importance for RF while we are here
            if "rf_gini_importance" in meta:
                rf_imp = pd.Series(meta["rf_gini_importance"]).sort_values(ascending=False).head(15)
                plt.figure(figsize=(10, 6))
                sns.barplot(x=rf_imp.values, y=rf_imp.index, palette='magma')
                plt.title('Random Forest Feature Importance (Gini)')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(FIG_DIR / "rf_feature_importance.png", dpi=300)
                plt.close()

            metrics = []
            for name, m in meta.items():
                if isinstance(m, dict) and "roc_auc" in m and "f1" in m:
                    metrics.append({"Algorithm": name, "ROC-AUC": m["roc_auc"], "F1-Score": m["f1"]})
            if metrics:
                df_metrics = pd.DataFrame(metrics).set_index("Algorithm")
                df_metrics.plot(kind='bar', figsize=(10, 5), colormap='Set1')
                plt.title('Algorithm Zoo: Performance Comparison')
                plt.ylabel('Score')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(FIG_DIR / "algorithm_comparison.png", dpi=300)
                plt.close()
        except Exception as e:
            print("Failed to process classification_metadata.joblib:", e)

    # 17. Cluster Profiles Heatmap
    cluster_file = MODELS_DIR / "cluster_results.json"
    if cluster_file.exists():
        with open(cluster_file, 'r') as f:
            clusters = json.load(f)
        profiles = clusters.get("cluster_profiles", {})
        if profiles:
            df_prof = pd.DataFrame(profiles).T
            numeric_prof = df_prof.select_dtypes(include=[np.number])
            if not numeric_prof.empty:
                plt.figure(figsize=(10, 6))
                sns.heatmap(numeric_prof, annot=True, cmap='YlGnBu', fmt=".2f")
                plt.title('K-Means Cluster Profiles (Centroids)')
                plt.ylabel('Cluster ID')
                plt.xlabel('Feature')
                plt.tight_layout()
                plt.savefig(FIG_DIR / "cluster_profiles.png", dpi=300)
                plt.close()

    print("All plots generated and saved to", FIG_DIR)

if __name__ == "__main__":
    generate_plots()
