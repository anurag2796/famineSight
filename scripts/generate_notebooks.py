import nbformat as nbf
from pathlib import Path
import os

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)

notebooks = [
    {
        "name": "01_data_exploration.ipynb",
        "title": "FamineSight: Data Exploration",
        "desc": "This notebook explores the raw humanitarian datasets (ACLED, CHIRPS, FSNAU, WFP, IPC) used in FamineSight."
    },
    {
        "name": "02_preprocessing.ipynb",
        "title": "FamineSight: Data Preprocessing",
        "desc": "This notebook demonstrates the data preprocessing pipeline including imputation, outlier clipping, and feature scaling."
    },
    {
        "name": "03_anomaly_detection.ipynb",
        "title": "FamineSight: Anomaly Detection",
        "desc": "This notebook explores Isolation Forest and Local Outlier Factor models for anomaly detection in humanitarian data."
    },
    {
        "name": "04_predictive_modeling.ipynb",
        "title": "FamineSight: Predictive Modeling",
        "desc": "This notebook trains and evaluates the Random Forest and XGBoost classifiers for predicting famine mortality risk."
    },
    {
        "name": "05_system_integration.ipynb",
        "title": "FamineSight: System Integration",
        "desc": "This notebook demonstrates the end-to-end integration of the system including prediction, analysis, and LLM narrative generation."
    }
]

for nb_info in notebooks:
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell(f"# {nb_info['title']}\n\n{nb_info['desc']}"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nimport os\n\n# Add project root to path\nsys.path.insert(0, os.path.abspath('..'))"),
        nbf.v4.new_markdown_cell("## Setup & Configuration"),
        nbf.v4.new_code_cell("from src.config import DATA_PROC, DATA_RAW\nprint(f'Data Directory: {DATA_PROC}')"),
        nbf.v4.new_markdown_cell("## Load Data"),
        nbf.v4.new_code_cell("# Example data loading\ntry:\n    df = pd.read_parquet(DATA_PROC / 'master_panel.parquet')\n    print(df.head())\nexcept Exception as e:\n    print(f'Could not load data: {e}')")
    ]
    
    with open(NOTEBOOKS_DIR / nb_info['name'], 'w') as f:
        nbf.write(nb, f)
        
print("Successfully generated 5 Jupyter Notebooks in the notebooks/ directory.")
