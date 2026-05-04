# Developer Guide

## Project Structure

The FamineSight project follows a modular architecture designed for maintainability and scalability:

```
FamineSight/
├── CLAUDE.md                 # Project specification and instructions
├── docker-compose.yml        # Docker orchestration
├── .env                      # Environment variables
├── .env.example              # Example environment variables
├── .dockerignore             # Docker ignore patterns
├── README.md                 # Project overview
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data files
│   └── synthetic/            # Synthetic data generation scripts
├── notebooks/                # Jupyter notebooks for analysis
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── data/                 # Data fetching and preprocessing
│   ├── analysis/             # Analytical modules
│   └── llm/                  # Large Language Model integration
├── backend/                  # FastAPI backend
│   ├── Dockerfile            # Backend Docker configuration
│   ├── requirements.txt      # Python dependencies
│   ├── main.py               # Main application
│   ├── routers/              # API route handlers
│   ├── schemas/              # Data models
│   ├── services/             # Business logic
│   └── tests/                # Test suite
├── frontend/                 # Streamlit frontend
│   ├── Dockerfile            # Frontend Docker configuration
│   ├── requirements.txt      # Python dependencies
│   └── app.py                # Main application
├── models/                   # Trained models
├── scripts/                  # Utility scripts
└── docs/                     # Documentation
    └── wiki/                 # Wiki documentation
```

## Configuration Management

### src/config.py

The configuration file centralizes all system settings:

```python
# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# File paths
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
MODELS_DIR = ROOT / "models"

# ACLED API credentials
ACLED_EMAIL = os.getenv("ACLED_EMAIL", "")
ACLED_PASSWORD = os.getenv("ACLED_PASSWORD", "")
ACLED_BASE_URL = "https://api.acleddata.com/acled/read"

# Thresholds
CDR_EMERGENCY_THRESHOLD = 1.0
DROUGHT_ANOMALY_THRESHOLD = -30.0
HIGH_CONFLICT_THRESHOLD = 10
PRICE_SPIKE_THRESHOLD = 150

# Jetson-specific configurations
RF_N_JOBS = int(os.getenv("RF_N_JOBS", "4"))
if RF_N_JOBS == -1:
    import warnings
    warnings.warn("RF_N_JOBS=-1 will cause OOM on Jetson AGX Orin. Setting to 4.", RuntimeWarning)
    RF_N_JOBS = 4

XGB_DEVICE = os.getenv("XGB_DEVICE", "cpu")
RANDOM_STATE = 42
LAG_MONTHS = [1, 2, 3]

# Feature column lists
CLIMATE_FEATURES = [
    "rainfall_anomaly_pct",
    "temperature_anomaly",
    "evapotranspiration_anomaly"
]

CONFLICT_FEATURES = [
    "conflict_events",
    "conflict_fatalities",
    "civilian_targeting_events"
]

MARKET_FEATURES = [
    "food_price_index",
    "inflation_rate",
    "exchange_rate"
]

IPC_FEATURES = [
    "ipc_phase1_pct",
    "ipc_phase2_pct",
    "ipc_phase3_pct",
    "ipc_phase4_pct",
    "ipc_phase5_pct"
]

ALL_FEATURES = CLIMATE_FEATURES + CONFLICT_FEATURES + MARKET_FEATURES + IPC_FEATURES

# Target and auxiliary variables
TARGET_COL = "crisis_label"
AUX_TARGETS = ["cdr_per_10k_per_day", "u5dr_per_10k_per_day"]

# Somalia districts and their p-codes
SOMALIA_DISTRICTS = [
    "Mogadishu", "Kismayo", "Baidoa", "Afgooye", "Luuq",
    "Hargeisa", "Berbera", "Galkayo", "El-Golea", "Gedo",
    "Jamaame", "Buurhakaba", "Dhuusamarreeb", "Mudug", "Sanaag",
    "Togdheer", "Sool", "Bay", "Galgaduud", "Hiiraan"
]

DISTRICT_PCODES = {
    "Mogadishu": "SO0001",
    "Kismayo": "SO0002",
    "Baidoa": "SO0003",
    "Afgooye": "SO0004",
    "Luuq": "SO0005",
    "Hargeisa": "SO0006",
    "Berbera": "SO0007",
    "Galkayo": "SO0008",
    "El-Golea": "SO0009",
    "Gedo": "SO0010",
    "Jamaame": "SO0011",
    "Buurhakaba": "SO0012",
    "Dhuusamarreeb": "SO0013",
    "Mudug": "SO0014",
    "Sanaag": "SO0015",
    "Togdheer": "SO0016",
    "Sool": "SO0017",
    "Bay": "SO0018",
    "Galgaduud": "SO0019",
    "Hiiraan": "SO0020"
}

# Association rule mining parameters
FP_MIN_SUPPORT = 0.05
APRIORI_MIN_CONFIDENCE = 0.6
APRIORI_MIN_LIFT = 1.2

# Clustering parameters
KMEANS_BEST_K = 4

# Classification parameters
RF_N_ESTIMATORS = 100
XGB_SCALE_POS_WEIGHT = 1.0
SMOTE_K_NEIGHBORS = 5

# Anomaly detection parameters
ISOFOREST_CONTAMINATION = 0.05
LOF_N_NEIGHBORS = 20
ZSCORE_THRESHOLD = 3.0

# LLM parameters
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:32b")
```

## Data Pipeline

### src/data/

The data module handles all data fetching and preprocessing:

#### acled_fetcher.py
Implements ACLED OAuth authentication and data fetching:
- OAuth token retrieval
- Automatic token refresh
- Pagination handling
- Rate limiting with exponential backoff
- Data cleaning and aggregation

#### wfp_fetcher.py
Fetches WFP food prices data:
- Direct CSV download from HDX
- Error handling for network issues
- Data processing and validation

#### preprocessor.py
Complete data preprocessing pipeline:
- Load and merge datasets
- Temporal sorting
- Missing value imputation
- Outlier clipping
- Lag feature engineering
- Feature scaling
- PCA dimensionality reduction
- Final data export

### scripts/fetch_data.py
Orchestrates the complete data fetching process:
- Real data fetching with fallback to synthetic
- Sequential data source processing
- Error handling and logging

## Analysis Modules

### src/analysis/

#### association.py
Association rule mining:
- FP-Growth and Apriori algorithms
- Binary transaction creation
- Rule filtering and sorting
- Sequential pattern mining

#### clustering.py
Clustering analysis:
- K-Means clustering with elbow method
- DBSCAN density-based clustering
- District profile computation
- Cluster labeling and interpretation

#### classification.py
Machine learning classification:
- Temporal train/validation/test splitting
- Random Forest with SMOTE
- XGBoost with CPU optimization
- Model evaluation and SHAP importance
- Feature engineering

#### anomaly.py
Anomaly detection:
- Isolation Forest
- Local Outlier Factor
- Z-score analysis
- Alert generation and prioritization

## LLM Integration

### src/llm/

#### client.py
Ollama client with:
- Async streaming interface
- Error handling and retry logic
- Timeout management
- Availability checking

#### prompts.py
Prompt engineering:
- System prompt with strict rules
- Prompt building functions
- Token limit management

#### guardrails.py
Content validation:
- Low risk + famine terms validation
- Probability mismatch checking
- Verification note enforcement
- Soft warnings for missing validation

## Backend API

### backend/

#### main.py
FastAPI application with:
- Lifecycle management
- CORS middleware
- Router registration
- Health check endpoint
- Model loading on startup

#### routers/
API endpoints:
- `/predict/mortality` - Mortality risk prediction
- `/analyze/rules` - Association rules
- `/analyze/clusters` - Cluster profiles
- `/anomaly/alerts` - Anomaly alerts
- `/narrative/generate` - AI narrative generation

#### schemas/
Data models:
- Input models for requests
- Output models for responses
- Validation with Pydantic v2

#### services/
Business logic:
- Model registry for managing ML models
- Inference service
- Data processing service

## Frontend Dashboard

### frontend/app.py
Streamlit dashboard with:
- Crisis predictor with visualization
- Vulnerability map with Folium
- Association rules display
- Anomaly alerts dashboard
- AI narrative generation
- Dark humanitarian theme

## Testing

### Unit Tests
All modules include unit tests covering:
- Functionality validation
- Error handling
- Edge cases
- Integration points

### Integration Tests
- End-to-end pipeline testing
- API endpoint testing
- Data flow validation
- Model performance verification

### Synthetic Data Testing
- Complete testing with synthetic data
- Fallback verification
- Performance benchmarking

## Deployment

### Docker Configuration

#### backend/Dockerfile
- ARM64 optimized
- GDAL system dependencies
- Python dependencies installation
- Proper resource limits

#### frontend/Dockerfile
- Streamlit deployment
- Required dependencies
- Port configuration

#### docker-compose.yml
- Multi-container orchestration
- Health checks
- Resource limits
- Network configuration

## Jetson Optimization

### ARM64 Specific Constraints
- `RF_N_JOBS=4` to prevent OOM
- `XGB_DEVICE="cpu"` for ARM64 compatibility
- GDAL installation in Dockerfile
- Memory-efficient processing

### Resource Management
- Docker resource limits
- Memory monitoring
- Performance profiling
- GPU/CPU utilization tracking

## Development Workflow

### Setting Up Development Environment
1. Clone repository
2. Create `.env` with credentials
3. Install dependencies
4. Run tests
5. Start development server

### Code Quality Standards
- Type hints for all functions
- Comprehensive docstrings
- PEP 8 compliance
- Unit tests for all modules
- Logging for debugging

### Version Control
- Git branching strategy
- Commit message conventions
- Pull request review process
- Release management

## Troubleshooting

### Common Issues
1. **Authentication Failures**: Check `.env` credentials
2. **Memory Issues**: Verify `RF_N_JOBS=4` setting
3. **Docker Build Failures**: Check GDAL dependencies
4. **API Rate Limiting**: System handles automatically
5. **Model Loading**: Check model files exist

### Debugging Techniques
1. **Log Analysis**: Check Docker logs
2. **Environment Verification**: Verify `.env` variables
3. **Component Testing**: Test individual modules
4. **Data Validation**: Check data quality
5. **Performance Profiling**: Monitor resource usage

## Performance Optimization

### Data Processing
- Efficient data structures
- Memory optimization
- Parallel processing where possible
- Caching for repeated operations

### Model Training
- Efficient hyperparameter tuning
- Feature selection
- Model compression
- Batch processing

### API Endpoints
- Response caching
- Database optimization
- Connection pooling
- Asynchronous processing

## Contributing

### Code Contribution Guidelines
1. Fork repository
2. Create feature branch
3. Write tests
4. Follow coding standards
5. Submit pull request

### Documentation Updates
- Update README.md for new features
- Add API documentation
- Update developer guides
- Add usage examples

## Support

### Contact Information
For support, please contact the development team or open an issue in the repository.

### Resources
- GitHub repository
- Issue tracker
- Community forums
- Documentation site