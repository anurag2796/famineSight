"""
FamineSight — Streamlit dashboard.

Pages map to sections of the syllabus used by this project.
Every figure is sourced from JSON/parquet
artifacts written by the training pipeline so this file stays free of
heavy ML deps.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROC = PROJECT_ROOT / "data" / "processed"

PLOT_BG = "rgba(0,0,0,0)"
FONT = dict(color="#e2e8f0", family="Inter, system-ui, sans-serif")

PAGE_INTROS = {
    "📋 Overview": (
        "This is the landing page for the dashboard. It summarizes the dataset footprint, "
        "shows the overall crisis trend over time, and confirms that the project covers the full pipeline from preprocessing to modeling."
    ),
    "🔍 Data Quality": (
        "This page checks whether the panel is complete enough to trust downstream modeling. "
        "The plots highlight missingness, outliers, and reporting coverage before any learning step begins."
    ),
    "🧪 Feature Engineering": (
        "This page shows how raw indicators are reshaped into model-ready features. "
        "Use it to see where discretization, scaling, lag structure, and feature ranking change the signal."
    ),
    "📊 EDA & High-Dim": (
        "This page gives a descriptive view of the panel and its lower-dimensional structure. "
        "It compares summary statistics, latent projections, and district-level variation to show how the data behaves overall."
    ),
    "🎯 Classification Lab": (
        "This page compares the main classifiers used for crisis prediction. "
        "The metrics and curves show whether the models are learning the boundary, calibrating well, and overfitting."
    ),
    "🤖 Algorithm Zoo": (
        "This page contains secondary classifier experiments that broaden the baseline beyond the core models. "
        "Use it to compare simpler learners against the main pipeline outputs."
    ),
    "🔗 Association": (
        "This page surfaces rule-based co-occurrence and sequential patterns. "
        "It helps explain which combinations of events and conditions tend to appear before crises."
    ),
    "🧬 Clustering Studio": (
        "This page groups districts by similar multivariate profiles. "
        "The plots show which areas cluster together and which dimensions separate them."
    ),
    "🚨 Anomaly Lab": (
        "This page highlights records that deviate sharply from the baseline distribution. "
        "Use the alert feed to inspect extreme values before they are treated as operational warnings."
    ),
    "📡 Predict & Narrative": (
        "This page combines the point prediction workflow with the generated situation report. "
        "The first tab tests a district-month input; the second turns the model output into a human-readable brief."
    ),
}

st.set_page_config(
    page_title="FamineSight — Data Mining Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp { background-color: #0b1220; color: #e2e8f0; }
section[data-testid="stSidebar"] { background-color: #111a2e; }
.stMetric { background: rgba(30,41,59,0.55); padding: 0.8rem 1rem; border-radius: 8px;
            border: 1px solid rgba(148,163,184,0.15); }
h1, h2, h3, h4 { color: #f1f5f9 !important; }
.small-muted { color: #94a3b8; font-size: 0.85rem; }
hr { border-color: rgba(148,163,184,0.15); }
[data-testid="stExpander"] { background: rgba(15,23,42,0.6); border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return None


@st.cache_data
def load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return None


@st.cache_resource
def load_joblib(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


VIZ = load_json(MODELS_DIR / "viz_payload.json") or {}
ASSOC = load_json(MODELS_DIR / "association_results.json") or {}
CLUSTER = load_json(MODELS_DIR / "cluster_results.json") or {}
ANOMALY = load_json(MODELS_DIR / "anomaly_results.json") or {}
META = load_joblib(MODELS_DIR / "classification_metadata.joblib") or {}
MASTER = load_parquet(DATA_PROC / "master_panel.parquet")
PCA_DF = load_parquet(DATA_PROC / "panel_pca.parquet")

# District lookup mapping (pcode -> human-friendly name)
_DISTRICT_CSV = PROJECT_ROOT / "data" / "raw" / "shapefiles" / "district_lookup.csv"
DISTRICT_MAP = {}
if _DISTRICT_CSV.exists():
    try:
        ddf = pd.read_csv(_DISTRICT_CSV)
        if "pcode" in ddf.columns and "name" in ddf.columns:
            DISTRICT_MAP = dict(zip(ddf["pcode"].astype(str), ddf["name"]))
    except Exception:
        DISTRICT_MAP = {}

# Reverse map for name -> pcode lookup
NAME_TO_PCODE = {v: k for k, v in DISTRICT_MAP.items()}

# Populate human-friendly district name where possible
if MASTER is not None and "pcode" in MASTER.columns:
    MASTER["district"] = MASTER["pcode"].astype(str).map(DISTRICT_MAP).fillna(MASTER["pcode"])
if PCA_DF is not None and "pcode" in PCA_DF.columns:
    PCA_DF["district"] = PCA_DF["pcode"].astype(str).map(DISTRICT_MAP).fillna(PCA_DF["pcode"])


def _theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT,
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(15,23,42,0.6)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.2)")
    return fig


def _missing(label: str, hint: str = ""):
    st.info(f"{label} not available. {hint}")


def _render_page_intro(page_name: str) -> None:
    intro = PAGE_INTROS.get(page_name)
    if intro:
        st.markdown(f"<p class='small-muted'>{intro}</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🌍 FamineSight")
st.sidebar.caption("Data-mining dashboard for Somalia famine early warning.")
st.sidebar.markdown(
    "<p class='small-muted'>Use the navigation to move from data quality checks to model diagnostics, clustering, anomaly detection, and the live prediction workflow. Each page pairs charts with short explanations so the visualizations can be read as a guided report, not just a gallery.</p>",
    unsafe_allow_html=True,
)

PAGES = [
    "📋 Overview",
    "🔍 Data Quality",
    "🧪 Feature Engineering",
    "📊 EDA & High-Dim",
    "🎯 Classification Lab",
    "🤖 Algorithm Zoo",
    "🔗 Association",
    "🧬 Clustering Studio",
    "🚨 Anomaly Lab",
    "📡 Predict & Narrative",
]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
if VIZ.get("generated_at"):
    st.sidebar.caption(f"Models last trained: {VIZ['generated_at'][:19].replace('T',' ')}")
else:
    st.sidebar.warning("viz_payload.json missing — run `python scripts/train_pipeline.py --synthetic`")

st.title(page)
_render_page_intro(page)

# ======================================================================
# Page: Overview
# ======================================================================
if page == "📋 Overview":
    if MASTER is None:
        _missing("Master panel", "Run the training pipeline.")
        st.stop()

    st.markdown(
        "<p class='small-muted'>Start here if you want the fast version of the project story. The metrics show the size and balance of the panel, the time-series chart shows how crisis months accumulate, and the coverage table confirms that the curriculum topics are implemented in the dashboard.</p>",
        unsafe_allow_html=True,
    )

    dq = VIZ.get("data_quality", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{dq.get('n_rows', len(MASTER)):,}")
    c2.metric("Districts", dq.get("n_districts", MASTER["district"].nunique()))
    c3.metric("Features", dq.get("n_features", "—"))
    c4.metric("Crisis prevalence", f"{MASTER['crisis_label'].mean():.1%}")

    st.markdown("### Crisis events over time")
    st.caption("This area chart shows how many district-months were labeled as crisis each month. Rising periods indicate broader pressure across Somalia, while dips suggest temporary relief or lower reporting intensity.")
    ts = MASTER.groupby("date")["crisis_label"].sum().reset_index()
    fig = px.area(ts, x="date", y="crisis_label",
                  color_discrete_sequence=["#ef4444"])
    fig.update_layout(title="Total district-months in crisis (per month)")
    st.plotly_chart(_theme(fig), use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("### Crisis prevalence by year")
        st.caption("Use this bar chart to compare crisis intensity across years. Taller bars mean a larger share of district-months in crisis during that year.")
        olap = VIZ.get("eda", {}).get("olap_year")
        if olap:
            df = pd.DataFrame(olap)
            f = px.bar(df, x="year", y="crisis_rate",
                       color="crisis_rate", color_continuous_scale="OrRd",
                       hover_data=["n_records"])
            st.plotly_chart(_theme(f), use_container_width=True)
        else:
            _missing("OLAP roll-up")

    with cc2:
        st.markdown("### Syllabus coverage")
        st.caption("This checklist shows which analysis modules are already represented in the dashboard and which parts of the pipeline are available in the generated artifacts.")
        cov = pd.DataFrame([
            ("Data preprocessing", "✅"),
            ("Exploring data + OLAP", "✅"),
            ("Decision tree + evaluation", "✅"),
            ("kNN / NB / SVM / MLP / RF / XGB", "✅"),
            ("Apriori / FP-Growth", "✅"),
            ("Sequential patterns", "✅"),
            ("K-Means / Hierarchical / DBSCAN", "✅"),
            ("GMM / MST", "✅"),
            ("Z-score / kth-NN / LOF / IForest", "✅"),
            ("PCA / SVD", "✅"),
        ], columns=["Topic", "Status"])
        st.dataframe(cov, hide_index=True, use_container_width=True)


# ======================================================================
# Page: Data Quality
# ======================================================================
elif page == "🔍 Data Quality":
    if MASTER is None:
        _missing("Master panel", "Run the training pipeline.")
        st.stop()

    dq = VIZ.get("data_quality", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{dq.get('n_rows', len(MASTER)):,}")
    c2.metric("Districts", dq.get("n_districts", MASTER["district"].nunique()))
    c3.metric("Features", dq.get("n_features", "—"))
    c4.metric("Crisis prevalence", f"{MASTER['crisis_label'].mean():.1%}")

    st.markdown("### Crisis events over time")
    ts = MASTER.groupby("date")["crisis_label"].sum().reset_index()
    fig = px.area(ts, x="date", y="crisis_label",
                  color_discrete_sequence=["#ef4444"])
    fig.update_layout(title="Total district-months in crisis (per month)")
    st.plotly_chart(_theme(fig), use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("### Crisis prevalence by year")
        olap = VIZ.get("eda", {}).get("olap_year")
        if olap:
            df = pd.DataFrame(olap)
            f = px.bar(df, x="year", y="crisis_rate",
                       color="crisis_rate", color_continuous_scale="OrRd",
                       hover_data=["n_records"])
            st.plotly_chart(_theme(f), use_container_width=True)
        else:
            _missing("OLAP roll-up")

    with cc2:
        st.markdown("### Syllabus coverage")
        cov = pd.DataFrame([
            ("Data preprocessing", "✅"),
            ("Exploring data + OLAP", "✅"),
            ("Decision tree + evaluation", "✅"),
            ("kNN / NB / SVM / MLP / RF / XGB", "✅"),
            ("Apriori / FP-Growth", "✅"),
            ("Sequential patterns", "✅"),
            ("K-Means / Hierarchical / DBSCAN", "✅"),
            ("GMM / MST", "✅"),
            ("Z-score / kth-NN / LOF / IForest", "✅"),
            ("PCA / SVD", "✅"),
        ], columns=["Topic", "Status"])
        st.dataframe(cov, hide_index=True, use_container_width=True)


# ==========================================================================
# Page: Feature Engineering
# ==========================================================================
elif page == "🧪 Feature Engineering":
    dq = VIZ.get("data_quality")
    if not dq:
        _missing("Data-quality payload"); st.stop()

    st.markdown("### Missingness per feature")
    st.caption("Features with long bars are the least complete and may require stronger imputation or more caution during modeling.")
    miss = pd.Series(dq["missing_pct"]).sort_values(ascending=True)
    fig = px.bar(x=miss.values, y=miss.index, orientation="h",
                 labels={"x": "% missing", "y": ""},
                 color=miss.values, color_continuous_scale="Reds")
    st.plotly_chart(_theme(fig, height=520), use_container_width=True)

    st.markdown("### Outlier rate (Tukey 1.5×IQR)")
    st.caption("Outliers are measured with a standard Tukey rule. High values do not always mean bad data, but they do show where the distribution is unusually heavy-tailed or volatile.")
    out = pd.Series(dq["outlier_pct"]).sort_values(ascending=True)
    fig = px.bar(x=out.values, y=out.index, orientation="h",
                 labels={"x": "% outliers", "y": ""},
                 color=out.values, color_continuous_scale="Oranges")
    st.plotly_chart(_theme(fig, height=520), use_container_width=True)

    cov = dq.get("coverage")
    if cov:
        st.markdown("### Data-availability coverage (district-fraction reporting)")
        st.caption("Darker cells mean more districts reported that feature in a given month. Sparse bands usually reveal gaps in source coverage rather than true zero values.")
        z = np.array(cov["matrix"])
        fig = go.Figure(go.Heatmap(
            z=z.T, x=cov["months"], y=cov["features"],
            colorscale="Viridis", colorbar_title="Coverage"))
        fig.update_layout(title="Month × feature coverage")
        st.plotly_chart(_theme(fig, height=560), use_container_width=True)


# ==========================================================================
# Page: EDA & High-Dim
# ==========================================================================
elif page == "📊 EDA & High-Dim":
    fe = VIZ.get("feature_engineering", {})

    disc = fe.get("discretization")
    if disc:
        st.markdown(f"### Discretization comparison — `{disc['feature']}`")
        st.caption("Same continuous feature binned three ways.")
        st.caption("The raw histogram shows the original shape, equal-width binning emphasizes absolute range, and equal-frequency binning shows how evenly the observations spread across bins.")
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Raw", "Equal-width (5 bins)", "Equal-frequency (5 bins)"))
        fig.add_trace(go.Histogram(x=disc["raw_hist"], marker_color="#60a5fa"), 1, 1)
        fig.add_trace(go.Bar(x=disc["equal_width"]["bins"], y=disc["equal_width"]["counts"],
                             marker_color="#f59e0b"), 1, 2)
        fig.add_trace(go.Bar(x=disc["equal_freq"]["bins"], y=disc["equal_freq"]["counts"],
                             marker_color="#10b981"), 1, 3)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_theme(fig), use_container_width=True)

    tx = fe.get("transformation")
    if tx:
        st.markdown(f"### Variable transformation — `{tx['feature']}`")
        st.caption("Raw vs. log vs. MinMax-scaled.")
        st.caption("Log transforms compress extreme values, while MinMax scaling places the feature on a common 0-1 scale for models that are sensitive to magnitude.")
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Raw", "log(1+x)", "MinMax → [0,1]"))
        fig.add_trace(go.Histogram(x=tx["raw"], marker_color="#60a5fa"), 1, 1)
        fig.add_trace(go.Histogram(x=tx["log"], marker_color="#a78bfa"), 1, 2)
        fig.add_trace(go.Histogram(x=tx["minmax"], marker_color="#34d399"), 1, 3)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_theme(fig), use_container_width=True)

    fs = VIZ.get("feature_selection")
    if fs and fs.get("features"):
        st.markdown("### Feature selection")
        st.caption("χ² and mutual-information rankings against `crisis_label`.")
        st.caption("Higher-ranked features are more informative about the target label and are the ones most likely to matter in the downstream classifiers.")
        df = pd.DataFrame({"feature": fs["features"], "chi2": fs["chi2"], "mi": fs["mutual_info"]})
        df = df.sort_values("mi", ascending=True).tail(20)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("χ² score", "Mutual information"))
        fig.add_trace(go.Bar(x=df["chi2"], y=df["feature"], orientation="h",
                             marker_color="#f87171"), 1, 1)
        fig.add_trace(go.Bar(x=df["mi"], y=df["feature"], orientation="h",
                             marker_color="#34d399"), 1, 2)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_theme(fig, height=540), use_container_width=True)

    acf = fe.get("lag_autocorr")
    if acf:
        st.markdown("### Lag autocorrelation")
        st.caption("Mean-across-districts autocorrelation per feature, lags 1-6 months.")
        st.caption("Strong positive autocorrelation means the feature persists over time, which is useful when the model needs temporal memory. Negative values suggest a more oscillating pattern.")
        fig = go.Figure(go.Heatmap(
            z=acf["matrix"], x=[f"lag {l}" for l in acf["lags"]],
            y=acf["features"], colorscale="RdBu", zmid=0,
            colorbar_title="ρ"))
        st.plotly_chart(_theme(fig), use_container_width=True)

    pca = VIZ.get("pca_diagnostics", {})
    if pca:
        st.markdown("### PCA scree / cumulative variance (Appendix B)")
        st.caption("The bars show variance explained by each component, while the line shows how quickly the representation captures the overall structure.")
        x = list(range(1, pca["n_components"] + 1))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=x, y=pca["explained_variance"],
                             name="explained variance", marker_color="#60a5fa"))
        fig.add_trace(go.Scatter(x=x, y=pca["cumulative"],
                                 name="cumulative", line=dict(color="#f59e0b", width=3)),
                      secondary_y=True)
        fig.update_yaxes(title="explained variance ratio", secondary_y=False)
        fig.update_yaxes(title="cumulative", secondary_y=True, range=[0, 1.05])
        fig.update_xaxes(title="component")
        st.plotly_chart(_theme(fig), use_container_width=True)


# ===========================================================================
# Page: EDA & High-Dim
# ===========================================================================
elif page == "📊 EDA & High-Dim":
    if MASTER is None:
        _missing("Master panel"); st.stop()
    eda = VIZ.get("eda", {})

    st.markdown("### Summary statistics")
    st.caption("This table is a quick numerical summary of the panel. Use it to spot skew, scale differences, and whether some variables need transformation before modeling.")
    desc = eda.get("describe")
    if desc:
        df = pd.DataFrame(desc["matrix"], index=desc["features"], columns=desc["columns"])
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(MASTER.describe().T, use_container_width=True)

    st.markdown("### Distribution snapshot (violins)")
    st.caption("Violin plots show the full distribution shape, not just the mean. Comparing the crisis and non-crisis groups helps reveal which variables separate the classes.")
    cols = [c for c in ("food_price_index", "rainfall_anomaly_pct", "ndvi_anomaly",
                        "conflict_fatalities", "ipc_phase4_pct") if c in MASTER.columns]
    if cols:
        long = MASTER[cols + ["crisis_label"]].melt(id_vars="crisis_label",
                                                    var_name="feature", value_name="value").dropna()
        fig = px.violin(long, x="feature", y="value", color="crisis_label",
                        box=True, points=False,
                        color_discrete_sequence=["#60a5fa", "#ef4444"])
        st.plotly_chart(_theme(fig, height=460), use_container_width=True)

    st.markdown("### Parallel coordinates")
    st.caption("Each line is a district-month. Parallel coordinates make it easier to see whether crisis rows follow a consistent multivariate pattern across several features at once.")
    par = eda.get("parallel")
    if par:
        df = pd.DataFrame(par["rows"], columns=par["features"])
        df["crisis"] = par["label"]
        fig = px.parallel_coordinates(
            df, color="crisis",
            color_continuous_scale=px.colors.diverging.RdYlBu_r,
            color_continuous_midpoint=0.5,
        )
        st.plotly_chart(_theme(fig, height=460), use_container_width=True)

    st.markdown("### t-SNE projection of PCA components")
    st.caption("The projection compresses many features into two or three dimensions so you can see whether crisis and non-crisis observations form visible neighborhoods.")
    ts = eda.get("tsne")
    if ts:
        hover = [DISTRICT_MAP.get(str(p), p) for p in ts.get("pcode", [])]
        if "z" in ts:
            fig = px.scatter_3d(x=ts["x"], y=ts["y"], z=ts["z"], color=ts["label"],
                                hover_name=hover, color_continuous_scale="OrRd", opacity=0.8,
                                labels={"x": "t-SNE 1", "y": "t-SNE 2", "z": "t-SNE 3", "color": "crisis"})
        else:
            fig = px.scatter(x=ts["x"], y=ts["y"], color=ts["label"], hover_name=hover,
                             color_continuous_scale="OrRd", opacity=0.75,
                             labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": "crisis"})
        st.plotly_chart(_theme(fig, height=520), use_container_width=True)

    st.markdown("### OLAP roll-up — district × year crisis rate")
    st.caption("This heatmap rolls the panel up by district and year. Darker cells mean a district spent more of that year in crisis.")
    if "date" in MASTER.columns and "pcode" in MASTER.columns:
        m = MASTER.copy()
        m["year"] = m["date"].dt.year
        pivot = m.pivot_table(index="pcode", columns="year",
                              values="crisis_label", aggfunc="mean")
        if not pivot.empty:
            # replace y-axis pcodes with human-friendly names where available
            y_labels = [DISTRICT_MAP.get(str(p), p) for p in pivot.index]
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.astype(str), y=y_labels,
                colorscale="OrRd", colorbar_title="rate"))
            fig.update_layout(title="District × year — fraction of months in crisis")
            st.plotly_chart(_theme(fig, height=620), use_container_width=True)


# ===========================================================================
# Page: Classification Lab
# ==========================================================================
elif page == "🎯 Classification Lab":
    curves = VIZ.get("classification_curves", {})
    sd = VIZ.get("split_demos", {})
    extra = VIZ.get("extra_models", {}).get("classifiers", {})

    if sd:
        st.markdown("### Splitting criteria")
        st.caption("Gini, Entropy and classification error as a function of class proportion p.")
        st.caption("These impurity curves explain why tree-based models choose certain split points: the best split is where the class mix becomes most pure.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sd["p"], y=sd["gini"], name="Gini", line=dict(color="#60a5fa")))
        fig.add_trace(go.Scatter(x=sd["p"], y=[e/2 for e in sd["entropy"]],
                                 name="Entropy/2", line=dict(color="#f59e0b")))
        fig.add_trace(go.Scatter(x=sd["p"], y=sd["classification_error"],
                                 name="Class. error", line=dict(color="#34d399")))
        fig.update_xaxes(title="P(class = 1)"); fig.update_yaxes(title="impurity")
        st.plotly_chart(_theme(fig), use_container_width=True)

    models = curves.get("models", {})
    all_models = {**models, **{k: v for k, v in extra.items() if isinstance(v, dict) and "roc" in v}}

    if all_models:
        st.markdown("### ROC overlay — every classifier")
        st.caption("ROC curves emphasize ranking quality. Curves that stay farther above the diagonal indicate better separation between crisis and non-crisis cases.")
        fig = go.Figure()
        palette = px.colors.qualitative.Set2
        for i, (name, d) in enumerate(all_models.items()):
            roc = d.get("roc")
            if roc:
                fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"],
                                         name=f"{name} (AUC={d.get('roc_auc', 0):.2f})",
                                         line=dict(color=palette[i % len(palette)], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#64748b"),
                                 showlegend=False))
        fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate")
        st.plotly_chart(_theme(fig), use_container_width=True)

        st.markdown("### Precision-Recall overlay")
        st.caption("Precision-recall is often more informative when crisis cases are rare, because it focuses on the model’s ability to find positives without flooding the alert stream.")
        fig = go.Figure()
        for i, (name, d) in enumerate(all_models.items()):
            pr = d.get("pr")
            if pr:
                fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"],
                                         name=f"{name} (AP={d.get('avg_precision', 0):.2f})",
                                         line=dict(color=palette[i % len(palette)], width=2)))
        fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision")
        st.plotly_chart(_theme(fig), use_container_width=True)

    if models:
        st.markdown("### Calibration & confusion (core models)")
        st.caption("Calibration shows whether predicted probabilities match observed frequencies, while the confusion matrix shows the types of mistakes each model makes.")
        cols = st.columns(min(3, len(models)))
        for i, (name, d) in enumerate(models.items()):
            with cols[i % len(cols)]:
                st.markdown(f"**{name}**")
                cal = d.get("calibration")
                if cal:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=cal["prob_pred"], y=cal["prob_true"],
                                             mode="lines+markers", line=dict(color="#60a5fa")))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#64748b"),
                                             showlegend=False))
                    fig.update_xaxes(title="predicted"); fig.update_yaxes(title="observed")
                    st.plotly_chart(_theme(fig, height=260), use_container_width=True, key=f"cal_{name}")
                cm = d.get("confusion_matrix")
                if cm:
                    fig = go.Figure(go.Heatmap(z=cm, x=["pred 0", "pred 1"], y=["true 0", "true 1"],
                                               colorscale="Blues", showscale=False,
                                               text=cm, texttemplate="%{text}"))
                    st.plotly_chart(_theme(fig, height=240), use_container_width=True, key=f"cm_{name}")

    if META and META.get("rf_gini_importance"):
        st.markdown("### Random-Forest feature importance (Gini)")
        st.caption("Higher bars mark variables that the random forest uses more often to split the data. This is a useful heuristic, but it should be read alongside the other diagnostics.")
        imp = pd.Series(META["rf_gini_importance"]).sort_values(ascending=True).tail(20)
        fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                     color=imp.values, color_continuous_scale="Plasma",
                     labels={"x": "importance", "y": ""})
        st.plotly_chart(_theme(fig, height=520), use_container_width=True)

    if META and META.get("dt_rules"):
        with st.expander("📜 Decision tree (depth ≤ 4) — extracted rules"):
            st.code(META["dt_rules"], language="text")


# ===========================================================================
# Page: Algorithm Zoo (extra classifiers)
# ==========================================================================
elif page == "🤖 Algorithm Zoo":
    extra = VIZ.get("extra_models", {}).get("classifiers", {})
    if not extra:
        extra_hint = "Run `python scripts/train_pipeline.py --synthetic` to generate Phase-B classifier results (kNN, Naive Bayes, SVM, MLP)."
        _missing("extra_models", extra_hint)
        st.stop()

    st.markdown("### Performance summary")
    st.caption("This table provides a compact comparison of the Phase-B classifiers. It is useful for seeing whether simpler models stay competitive with the main pipeline.")
    rows = []
    for k, d in extra.items():
        if not isinstance(d, dict) or "error" in d:
            continue
        rows.append({"model": d.get("name", k),
                     "precision": round(d.get("precision", 0), 3),
                     "recall": round(d.get("recall", 0), 3),
                     "F1": round(d.get("f1", 0), 3),
                     "ROC-AUC": round(d.get("roc_auc", 0), 3),
                     "AP": round(d.get("avg_precision", 0), 3)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if extra.get("knn", {}).get("k_curve"):
        st.markdown("### kNN — accuracy vs. k")
        st.caption("This curve shows how neighborhood size changes the bias-variance tradeoff. Very small k can overfit noise, while very large k can smooth away useful signal.")
        kc = pd.DataFrame(extra["knn"]["k_curve"])
        fig = px.line(kc, x="k", y="accuracy", markers=True,
                      color_discrete_sequence=["#60a5fa"])
        st.plotly_chart(_theme(fig, height=320), use_container_width=True)

    mlp = extra.get("mlp", {})
    if mlp.get("loss_curve"):
        st.markdown("### MLP — training loss curve")
        st.caption("A downward loss curve means the network is fitting the training data. Flattening or oscillation can point to slower convergence or instability.")
        fig = px.line(y=mlp["loss_curve"], labels={"x": "epoch", "y": "loss"},
                      color_discrete_sequence=["#a78bfa"])
        st.plotly_chart(_theme(fig, height=320), use_container_width=True)

    svm = extra.get("svm", {})
    if svm.get("n_support"):
        st.markdown("### SVM support-vector counts")
        st.caption("Support vectors are the training points that define the decision boundary. More support vectors usually mean a more complex boundary.")
        sv = pd.DataFrame({"class": [f"class {i}" for i in range(len(svm['n_support']))],
                           "support_vectors": svm["n_support"]})
        fig = px.bar(sv, x="class", y="support_vectors",
                     color_discrete_sequence=["#f59e0b"])
        st.plotly_chart(_theme(fig, height=320), use_container_width=True)


# ===========================================================================
# Page: Association
# ==========================================================================
elif page == "🔗 Association":
    stats = ASSOC.get("stats", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FP-Growth rules", stats.get("fp_n_rules", 0))
    c2.metric("Apriori rules", stats.get("apriori_n_rules", 0))
    c3.metric("Sequential patterns", stats.get("seq_n_patterns", 0))
    c4.metric("Crisis prevalence", f"{stats.get('crisis_prevalence', 0):.1%}")

    algo = st.radio("Algorithm", ["FP-Growth", "Apriori"], horizontal=True)
    st.caption("Association rules are read as if-then statements. High confidence means the rule is reliable, high lift means it appears more often than chance, and support shows how common it is.")
    rules = ASSOC.get("fpgrowth_rules" if algo == "FP-Growth" else "apriori_rules", [])
    if rules:
        df = pd.DataFrame(rules)
        for col in ("antecedents", "consequents"):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        st.markdown("### Lift × Confidence × Support")
        st.caption("Use this chart to find rules that are both strong and common enough to matter operationally. The data table below lists the highest-lift rules first.")
        # Prefer 3D interactive scatter if support dimension available
        if all(c in df.columns for c in ("confidence", "lift", "support")):
            hover = [h for h in ("antecedents", "consequents") if h in df.columns]
            fig = px.scatter_3d(df, x="confidence", y="lift", z="support",
                                color="lift", size="support",
                                hover_data=hover, color_continuous_scale="RdYlGn")
        else:
            fig = px.scatter(df, x="confidence", y="lift", size="support",
                             color="lift", color_continuous_scale="RdYlGn",
                             hover_data=[c for c in ("antecedents", "consequents") if c in df.columns])
        st.plotly_chart(_theme(fig), use_container_width=True)
        st.dataframe(df.sort_values("lift", ascending=False).head(25),
                     use_container_width=True, hide_index=True)
    else:
        st.info(f"No {algo} rules at the configured support/confidence thresholds. "
                "Try lowering FP_MIN_SUPPORT in src/config.py.")

    st.markdown("### Sequential patterns")
    st.caption("Sequential patterns capture time-ordered combinations, such as drought followed by conflict and then mortality pressure.")
    seqs = ASSOC.get("sequential_patterns", [])
    if seqs:
        df = pd.DataFrame(seqs).sort_values("frequency", ascending=False).head(20)
        fig = px.bar(df, x="frequency", y="pattern", orientation="h",
                     color="frequency", color_continuous_scale="Plasma")
        st.plotly_chart(_theme(fig, height=520), use_container_width=True)
        st.caption("Symbols: D=Drought · C=Conflict · P=Price spike · M=Mortality crisis")


# ===========================================================================
# Page: Clustering Studio
# ==========================================================================
elif page == "🧬 Clustering Studio":
    km = CLUSTER.get("kmeans_metrics", {})
    db = CLUSTER.get("dbscan_summary", {})
    profiles = CLUSTER.get("district_profiles", [])
    extras = VIZ.get("clustering_extras", {})
    extra_models = VIZ.get("extra_models", {}).get("clustering", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("K-Means silhouette", f"{km.get('silhouette', 0):.3f}")
    c2.metric("Davies-Bouldin", f"{km.get('davies_bouldin', 0):.3f}")
    c3.metric("Calinski-Harabasz", f"{km.get('calinski_harabasz', 0):.1f}")
    c4.metric("DBSCAN clusters", db.get("n_clusters", 0))

    elbow = extras.get("elbow")
    if elbow:
        st.markdown("### Elbow & silhouette per k")
        st.caption("Elbow helps spot diminishing returns in SSE, while silhouette measures how cleanly separated the clusters are at each k.")
        df = pd.DataFrame(elbow)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df["k"], y=df["sse"], name="SSE",
                                 line=dict(color="#60a5fa", width=3), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=df["k"], y=df["silhouette"], name="silhouette",
                                 line=dict(color="#f59e0b", width=3), mode="lines+markers"),
                      secondary_y=True)
        fig.update_xaxes(title="k")
        st.plotly_chart(_theme(fig, height=380), use_container_width=True)

    dendro = extras.get("dendrogram")
    if dendro:
        st.markdown("### Agglomerative dendrogram (Ward linkage)")
        st.caption("Branches that merge late are more dissimilar. The tree is useful for seeing how districts group together before a final cluster choice is made.")
        try:
            Z = np.array(dendro["linkage"])
            ddata = dendrogram(Z, no_plot=True, labels=dendro["labels"])
            fig = go.Figure()
            for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                         line=dict(color="#94a3b8", width=1),
                                         hoverinfo="none", showlegend=False))
            tick_x = list(range(5, 5 + 10 * len(ddata["ivl"]), 10))
            fig.update_layout(xaxis=dict(tickmode="array", tickvals=tick_x,
                                         ticktext=ddata["ivl"], tickangle=-60),
                              yaxis_title="distance")
            st.plotly_chart(_theme(fig, height=520), use_container_width=True)
        except Exception as e:
            st.warning(f"Dendrogram render failed: {e}")

    if profiles and km.get("labels"):
        st.markdown("### Cluster centroid heatmap (GMM comparison)")
        st.caption("Each row is a cluster centroid. Strong contrasts across columns show which variables most clearly differentiate the district groups.")
        df = pd.DataFrame(profiles).copy()
        df["kmeans"] = km["labels"]
        feats = [c for c in df.columns if c not in ("pcode", "district", "kmeans")
                 and pd.api.types.is_numeric_dtype(df[c])]
        cent = df.groupby("kmeans")[feats].mean()
        names = km.get("cluster_names", {})
        cent.index = [names.get(str(i), f"cluster {i}") for i in cent.index]
        fig = go.Figure(go.Heatmap(z=cent.values, x=cent.columns, y=cent.index,
                                   colorscale="RdBu", zmid=cent.values.mean(),
                                   colorbar_title="mean (scaled)"))
        st.plotly_chart(_theme(fig, height=420), use_container_width=True)

    gmm = extra_models.get("gmm")
    if gmm:
        st.markdown("### GMM — BIC / AIC vs. k")
        st.caption("Lower BIC/AIC suggests a better tradeoff between fit and complexity. The preferred k is often near the minimum, subject to interpretability.")
        df = pd.DataFrame(gmm["k_curve"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["k"], y=df["bic"], name="BIC",
                                 line=dict(color="#60a5fa", width=3), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=df["k"], y=df["aic"], name="AIC",
                                 line=dict(color="#f59e0b", width=3), mode="lines+markers"))
        fig.update_xaxes(title="components")
        st.plotly_chart(_theme(fig, height=320), use_container_width=True)

    mst = extra_models.get("mst")
    if mst and mst.get("edges"):
        st.markdown("### Minimum spanning tree of districts")
        st.caption("Edge weight = euclidean distance over scaled features. Heavy edges indicate dissimilar neighbors.")
        edges_df = pd.DataFrame(mst["edges"]).sort_values("weight").head(60)
        st.dataframe(edges_df, use_container_width=True, hide_index=True)

    if profiles and PCA_DF is not None and "pca_comp_1" in PCA_DF.columns:
        st.markdown("### District projection on PC1-PC2 (colored by K-Means)")
        st.caption("This projection shows whether cluster membership is visible in a lower-dimensional space. Nearby points are more similar after PCA compression.")
        df = pd.DataFrame(profiles)
        df["kmeans"] = km.get("labels", [0] * len(df))
        pca_means = (PCA_DF[["pcode", "pca_comp_1", "pca_comp_2"]]
                     .groupby("pcode").mean().reset_index())
        merged = df.merge(pca_means, on="pcode", how="inner")
        if not merged.empty:
            names = km.get("cluster_names", {})
            merged["cluster"] = merged["kmeans"].apply(lambda i: names.get(str(int(i)), str(i)))
            merged["district"] = merged["pcode"].astype(str).map(DISTRICT_MAP).fillna(merged["pcode"])
            if "pca_comp_3" in merged.columns:
                # 3D interactive scatter when 3 PCA comps available
                fig = go.Figure()
                for cl in merged["cluster"].unique():
                    sub = merged[merged["cluster"] == cl]
                    fig.add_trace(go.Scatter3d(
                        x=sub["pca_comp_1"], y=sub["pca_comp_2"], z=sub["pca_comp_3"],
                        mode='markers', name=str(cl), marker=dict(size=5),
                        text=sub["district"], hoverinfo='text'))
                fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
                st.plotly_chart(_theme(fig, height=520), use_container_width=True)
            else:
                fig = px.scatter(merged, x="pca_comp_1", y="pca_comp_2",
                                 color="cluster", hover_name="district",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(_theme(fig, height=460), use_container_width=True)


# ===========================================================================
# Page: Anomaly Lab
# ==========================================================================
elif page == "🚨 Anomaly Lab":
    c1, c2, c3 = st.columns(3)
    c1.metric("Total anomalies", ANOMALY.get("total_anomalies", 0))
    c2.metric("Critical alerts", ANOMALY.get("critical_count", 0))
    c3.metric("Anomaly rate", f"{ANOMALY.get('anomaly_rate', 0):.1%}")

    anom = VIZ.get("anomaly_extras", {})
    if anom.get("zscore"):
        st.markdown("### Z-score distributions")
        st.caption("Threshold lines at ±3σ.")
        st.caption("Values far beyond the threshold are unusual relative to the rest of the panel and are therefore candidates for anomaly review.")
        cols = st.columns(len(anom["zscore"]))
        for i, (feat, d) in enumerate(anom["zscore"].items()):
            with cols[i]:
                fig = px.histogram(x=d["values"], nbins=40,
                                   color_discrete_sequence=["#60a5fa"])
                fig.add_vline(x=3, line_dash="dash", line_color="#ef4444")
                fig.add_vline(x=-3, line_dash="dash", line_color="#ef4444")
                fig.update_xaxes(title=feat); fig.update_yaxes(title="")
                st.plotly_chart(_theme(fig, height=320), use_container_width=True, key=f"z_{feat}")

    if anom.get("kth_nn"):
        st.markdown("### k-th nearest-neighbor distance")
        st.caption("Large distances mean the observation sits far from its nearest neighbors in feature space, which is a classic sign of local isolation.")
        d = anom["kth_nn"]
        fig = px.histogram(x=d["distances"], nbins=60,
                           color_discrete_sequence=["#a78bfa"])
        fig.update_xaxes(title=f"distance to {d['k']}-th NN")
        st.plotly_chart(_theme(fig, height=320), use_container_width=True)

    st.markdown("### Alert feed")
    st.caption("This table consolidates the strongest anomaly signals into an operational queue. Use the severity filter to focus on the cases that deserve review first.")
    sev = st.multiselect("Severity", ["CRITICAL", "WARNING"],
                         default=["CRITICAL", "WARNING"])
    alerts = [a for a in ANOMALY.get("alerts", []) if a.get("severity") in sev]
    if alerts:
        df = pd.DataFrame(alerts)
        if "anomaly_flags" in df.columns:
            df["triggered_by"] = df["anomaly_flags"].apply(
                lambda d: ", ".join(k for k, v in (d or {}).items() if v))
            df = df.drop(columns=["anomaly_flags"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts at the chosen severity.")

    noise = VIZ.get("extra_models", {}).get("clustering", {}).get("dbscan_noise_pcodes")
    if noise:
        st.markdown("### Clustering-based anomalies — DBSCAN noise points")
        st.code(", ".join(noise))


# ===========================================================================
# Page: Predict & Narrative
# ===========================================================================
elif page == "📡 Predict & Narrative":
    tab1, tab2 = st.tabs(["🎯 Predict", "📝 Narrative"])

    with tab1:
        st.caption("Enter a district-month scenario and the API returns a crisis probability. The gauge below turns that probability into a simple visual risk band.")
        c1, c2 = st.columns(2)
        with c1:
            district_pcode = st.text_input("District (pcode or name)", value="SO2501")
            rainfall = st.number_input("Rainfall anomaly (%)", value=0.0, format="%.2f")
            conflict = st.number_input("Conflict fatalities", value=0, min_value=0)
        with c2:
            food_price = st.number_input("Food price index", value=100.0, format="%.2f")
            ipc4 = st.number_input("IPC Phase 4 (%)", value=0.0, min_value=0.0, max_value=100.0)
            ipc5 = st.number_input("IPC Phase 5 (%)", value=0.0, min_value=0.0, max_value=100.0)

        if st.button("Predict", type="primary"):
            dp = district_pcode
            # allow entering either pcode or human-friendly name
            if dp in NAME_TO_PCODE:
                dp = NAME_TO_PCODE[dp]
            payload = {"district_pcode": dp,
                       "rainfall_anomaly_pct": rainfall,
                       "conflict_fatalities": conflict,
                       "food_price_index": food_price,
                       "ipc_phase4_pct": ipc4,
                       "ipc_phase5_pct": ipc5}
            try:
                r = requests.post(f"{API_URL}/predict/mortality", json=payload,
                                  headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    risk = data.get("risk_level", "Unknown")
                    prob = data.get("probability", 0.0)
                    cc1, cc2 = st.columns(2)
                    cc1.metric("Risk level", risk)
                    cc2.metric("Crisis probability", f"{prob:.1%}")
                    try:
                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=prob * 100,
                                gauge=dict(
                                    axis=dict(range=[0, 100]),
                                    bar=dict(color="#ef4444"),
                                    steps=[
                                        dict(range=[0, 30], color="rgba(16,185,129,0.2)"),
                                        dict(range=[30, 70], color="rgba(245,158,11,0.2)"),
                                        dict(range=[70, 100], color="rgba(239,68,68,0.2)")
                                    ],
                                ),
                            )
                        )
                        st.plotly_chart(_theme(fig, height=320), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Gauge unavailable: {e}")
                else:
                    st.error(f"API {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Backend not reachable: {e}")

    with tab2:
        st.caption("The narrative tab turns the prediction payload into a short situation report. It is intended to be readable by non-technical users after the model has scored the case.")
        district = st.text_input("District (pcode or name)", value="SO2501", key="narr_pcode")
        if st.button("Generate situation report", type="primary"):
            d = district
            if d in NAME_TO_PCODE:
                d = NAME_TO_PCODE[d]
            payload = {"prediction": {"district_pcode": d}, "alerts": [], "rules": {}}
            try:
                r = requests.post(f"{API_URL}/narrative/generate", json=payload,
                                  headers=HEADERS, stream=True, timeout=180)
                if r.status_code == 200:
                    placeholder = st.empty()
                    text = ""
                    for chunk in r.iter_content(chunk_size=512):
                        if chunk:
                            text += chunk.decode("utf-8", errors="ignore")
                            placeholder.markdown(text + "▌")
                    placeholder.markdown(text)
                    st.warning("⚠️ AI-generated. Verify with field teams before action.")
                else:
                    st.error(f"API {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Backend not reachable: {e}")
