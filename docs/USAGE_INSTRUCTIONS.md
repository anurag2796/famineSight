# FamineSight Usage Instructions

## System Overview

FamineSight is a humanitarian data mining system designed to predict hunger-related mortality in Somalia. It integrates multiple data sources and employs advanced analytics to provide early warning capabilities.

## System Architecture

```
FamineSight/
├── src/                      # Source code
│   ├── data/                 # Data fetching and preprocessing
│   ├── analysis/             # Analytical modules
│   ├── llm/                  # LLM integration (Ollama + Groq)
│   └── config.py             # Configuration management
├── backend/                  # FastAPI backend
├── frontend/                 # Streamlit frontend
└── data/                     # Data storage
    ├── raw/                  # Raw data files
    └── processed/            # Processed data files
```

## Configuration

### Environment Variables (.env file)

```bash
# ACLED API credentials
ACLED_EMAIL=your_email@organization.org
ACLED_PASSWORD=your_secure_password

# LLM Configuration
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=mistral:7b
GROQ_API_KEY=your_groq_api_key_here     # Optional - for development
GROQ_MODEL=llama3-8b-8192               # Optional - for development

# System settings
RF_N_JOBS=4
XGB_DEVICE=cpu
BACKEND_URL=http://backend:8000
```

## Running the System

### 1. Development Mode (with smaller LLM)

```bash
# Set up environment with development model
echo "OLLAMA_MODEL=mistral:7b" >> .env

# Start the system
docker compose up -d

# Test the system
curl -s http://localhost:8000/health
```

### 2. Production Mode (with full model)

```bash
# Set up environment with production model
echo "OLLAMA_MODEL=qwen3:32b" >> .env

# Start the system
docker compose up -d
```

### 3. Development with Groq (for testing)

```bash
# Set up environment with Groq
echo "GROQ_API_KEY=your_groq_key_here" >> .env
echo "GROQ_MODEL=llama3-8b-8192" >> .env

# System will automatically use Groq for LLM operations when available
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

## LLM Integration

### Hybrid Client Architecture
The system uses a hybrid client that automatically chooses between:
1. **Local Ollama** (primary - secure, offline)
2. **Groq API** (secondary - for development/testing)

### Usage Examples

#### Using Ollama (default)
```python
from src.llm.client import hybrid_client

# This will use Ollama when available
async for chunk in hybrid_client.stream("Hello, how are you?"):
    print(chunk, end="", flush=True)
```

#### Using Groq (when configured)
```python
# Set GROQ_API_KEY in .env
# System will automatically use Groq when available
```

## Testing

### Run System Tests
```bash
# Test with synthetic data
python test_integration_end_to_end.py

# Test LLM integration
python test_groq_integration.py
```

### Docker Testing
```bash
# Test with Docker
docker compose -f docker-compose.dev.yml up -d
```

## Production Deployment

### Prerequisites
- Jetson AGX Orin with sufficient RAM (60+ GB)
- Qwen3:32b LLM (via Ollama)
- Docker and Docker Compose
- ACLED API credentials

### Deployment Steps
1. **Configure environment**
   ```bash
   echo "OLLAMA_MODEL=qwen3:32b" >> .env
   ```

2. **Build and start**
   ```bash
   docker compose build
   docker compose up -d
   ```

3. **Verify deployment**
   ```bash
   curl -s http://localhost:8000/health
   ```

## Troubleshooting

### Common Issues

1. **ACLED Authentication**
   ```bash
   # Check credentials
   grep ACLED .env
   ```

2. **LLM Connectivity**
   ```bash
   # Check Ollama availability
   docker compose exec backend curl -s http://host.docker.internal:11434/api/tags
   ```

3. **Network Issues**
   ```bash
   # Test connectivity
   ping acleddata.com
   ```

### Error Handling
The system implements graceful fallback:
- If ACLED API fails → uses synthetic data
- If LLM fails → uses fallback mechanisms
- All errors logged for monitoring

## Security

### Data Privacy
- All processing done locally on Jetson
- No external data transmission
- Secure credential handling
- Environment variable storage only

### API Security
- Credentials in `.env` file (600 permissions)
- No hardcoded credentials
- Secure token management

## Performance Optimization

### Jetson Constraints
- `RF_N_JOBS=4` to prevent OOM
- `XGB_DEVICE="cpu"` for ARM64 compatibility
- Efficient memory usage
- Proper Docker resource limits

### Resource Monitoring
```bash
# Monitor system resources
htop
docker stats
```

## Future Enhancements

### Planned Improvements
1. **Enhanced Groq Integration** - Better error handling and fallback
2. **Multi-model Support** - Support for multiple LLMs
3. **Advanced Analytics** - Additional machine learning models
4. **Cloud Integration** - Optional cloud-based processing
5. **Mobile App** - Mobile interface for field workers

## Support

For support, please contact the development team or open an issue in the repository.

## License

MIT License - see LICENSE file for details.