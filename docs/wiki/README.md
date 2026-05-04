# FamineSight Documentation

## Overview

FamineSight is a comprehensive humanitarian data mining system designed to predict hunger-related mortality in Somalia. The system integrates multiple data sources and employs advanced analytics to provide early warning capabilities for humanitarian response teams.

## System Architecture

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

## Key Features

### 1. Data Integration
- ACLED API integration with OAuth authentication
- WFP food price data fetching
- Synthetic data generation for testing
- Multi-source data fusion

### 2. Analytical Capabilities
- Association rule mining (FP-Growth, Apriori)
- Clustering analysis (K-Means, DBSCAN)
- Machine learning classification (Random Forest, XGBoost)
- Anomaly detection (Isolation Forest, LOF)
- LLM-based narrative generation

### 3. Platform Optimization
- ARM64/Jetson AGX Orin optimized
- Memory-safe configurations
- Docker-based deployment
- Real-time processing capabilities

## Getting Started

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Jetson AGX Orin with sufficient RAM (60+ GB)
- Qwen3:32b LLM (via Ollama)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd faminesight
```

2. **Set up environment variables**
```bash
# Copy example file and add your credentials
cp .env.example .env
# Edit .env to add your ACLED credentials
```

3. **Install dependencies**
```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

4. **Start the system**
```bash
docker compose up --build
```

## ACLED API Integration

### Authentication Process
FamineSight uses OAuth 2.0 authentication with the ACLED API:

1. **Get OAuth Token**:
   - POST to `https://acleddata.com/oauth/token`
   - Parameters: username (email), password, grant_type="password", client_id="acled", scope="authenticated"

2. **Use Token**:
   - Include in headers: `Authorization: Bearer <token>`
   - Token valid for 24 hours

3. **Token Refresh**:
   - Automatically refreshes on 401 errors
   - Handles token expiration gracefully

### Environment Setup
```bash
# In .env file
ACLED_EMAIL=your_email@organization.org
ACLED_PASSWORD=your_secure_password
```

## Data Pipeline

### Data Sources
1. **ACLED Conflict Data**: Somalia conflict events (2010-2024)
2. **WFP Food Prices**: Food price indices for Somalia
3. **CHIRPS Rainfall**: Climate data for Somalia
4. **FSNAU Mortality**: Famine-related mortality data
5. **IPC Phases**: Integrated Food Security Phase classification

### Processing Steps
1. **Data Fetching**: Real data or synthetic fallback
2. **Preprocessing**: Merging, imputation, scaling
3. **Feature Engineering**: Lag features, rolling statistics
4. **Dimensionality Reduction**: PCA for efficient processing
5. **Temporal Sorting**: Ensures chronological data flow

## Model Architecture

### Classification Models
- **Random Forest**: Ensemble method with SMOTE for class balancing
- **XGBoost**: Gradient boosting with CPU optimization for ARM64

### Clustering
- **K-Means**: 4-cluster analysis for district vulnerability profiles
- **DBSCAN**: Density-based clustering for conflict epicenters

### Anomaly Detection
- **Isolation Forest**: Tree-based anomaly detection
- **Local Outlier Factor**: Density-based anomaly detection
- **Z-Score Analysis**: Statistical anomaly detection

## API Endpoints

### Prediction
- `POST /predict/mortality` - Predict mortality risk for a district

### Analysis
- `GET /analyze/rules` - Get association rules
- `GET /analyze/clusters` - Get cluster profiles

### Anomaly Detection
- `GET /anomaly/alerts` - Get anomaly alerts

### Narrative Generation
- `POST /narrative/generate` - Generate AI situation report

## Deployment

### Docker Configuration
The system uses Docker Compose for deployment:

1. **Backend Service**: FastAPI API server
2. **Frontend Service**: Streamlit dashboard
3. **Networking**: Proper host-gateway mapping for Ollama connectivity

### Resource Requirements
- **CPU**: 12+ cores (Jetson AGX Orin)
- **Memory**: 60+ GB RAM (shared with GPU)
- **Storage**: 50-70 GB SSD
- **GPU**: Jetson AGX Orin 64GB

## Development Guidelines

### Code Structure
- Modular design with clear separation of concerns
- Configuration-driven approach
- Comprehensive error handling
- Logging for debugging and monitoring

### Jetson Optimization
- `RF_N_JOBS=4` to prevent OOM
- `XGB_DEVICE="cpu"` for ARM64 compatibility
- GDAL system dependencies in Dockerfile
- All constraints documented in CLAUDE.md

### Testing
- Unit tests for each module
- Integration tests for data pipeline
- End-to-end tests for API
- Synthetic data for testing without real API access

## Troubleshooting

### Common Issues
1. **API Authentication**: Check credentials in `.env`
2. **Docker Build**: Ensure GDAL dependencies are installed
3. **Memory Issues**: Verify `RF_N_JOBS=4` setting
4. **LLM Connectivity**: Ensure Ollama is running on host

### Debugging
- Check logs in Docker containers
- Use `docker compose logs` for detailed error information
- Verify environment variables are loaded correctly
- Test individual components separately

## Contributing

### Code Style
- Follow PEP 8 conventions
- Use type hints
- Include docstrings for all functions
- Write comprehensive tests

### Development Process
1. Fork the repository
2. Create feature branch
3. Make changes
4. Write tests
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

For support, please contact the development team or open an issue in the repository.