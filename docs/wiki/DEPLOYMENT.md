# Deployment Guide

## Overview

This document provides comprehensive instructions for deploying the FamineSight system in production environments, including Jetson AGX Orin platforms and other ARM64 systems.

## System Requirements

### Hardware Requirements
- **CPU**: ARM64 processor (Jetson AGX Orin 64GB recommended)
- **Memory**: 60+ GB RAM (shared with GPU)
- **Storage**: 50-70 GB SSD (with 100 GB total available)
- **GPU**: Jetson AGX Orin GPU (for accelerated processing)
- **Network**: Internet connectivity for API access and Ollama

### Software Requirements
- **Operating System**: Ubuntu 22.04 (JetPack 6.x)
- **Docker**: Docker Engine with Docker Compose v2
- **Python**: 3.11
- **Ollama**: Qwen3:32b model (via Ollama)
- **ACLED API**: Valid credentials for data access

## Deployment Steps

### 1. Environment Setup

#### Clone Repository
```bash
git clone <repository-url>
cd faminesight
```

#### Set Up Environment Variables
```bash
# Copy example file
cp .env.example .env

# Edit .env to add your credentials
echo "ACLED_EMAIL=your_email@organization.org" >> .env
echo "ACLED_PASSWORD=your_secure_password" >> .env
```

#### Install System Dependencies
```bash
# Install GDAL dependencies (required for ARM64)
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev libgeos-dev libproj-dev build-essential

# Install Python dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Docker Configuration

#### Build Docker Images
```bash
# Build all services
docker compose build

# Or build specific services
docker compose build backend
docker compose build frontend
```

#### Configure Docker Resources
The `docker-compose.yml` file includes resource limits:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Backend service
    # Frontend service limited to 2G
```

### 3. Model Preparation

#### Download Required Models
```bash
# Start Ollama service
docker compose up -d backend

# Ensure Ollama is running and download Qwen3:32b
# Run this on the host machine:
ollama run qwen3:32b "hello"
```

#### Verify Model Availability
```bash
# Check if model is available
docker compose exec backend ollama list
```

### 4. Data Pipeline Execution

#### Fetch Data
```bash
# Fetch real data (will use ACLED credentials from .env)
python scripts/fetch_data.py

# Or force synthetic data for testing
python scripts/fetch_data.py --synthetic
```

#### Verify Data Quality
```bash
# Check that data was fetched
ls data/raw/acled/
ls data/raw/wfp/
ls data/raw/fsnau/
ls data/raw/ipc/
ls data/raw/shapefiles/
```

### 5. System Startup

#### Start All Services
```bash
# Start the complete system
docker compose up -d

# Check service status
docker compose ps
```

#### Verify Services
```bash
# Check backend health
curl -s http://localhost:8000/health

# Check frontend availability
curl -s http://localhost:8001  # Streamlit port
```

## Production Configuration

### Security Considerations

#### Environment Variables
- Store all credentials in `.env` file
- Set proper file permissions: `chmod 600 .env`
- Never commit credentials to version control

#### Network Security
- Use HTTPS for all communications
- Configure firewall rules
- Implement proper access controls

#### Data Security
- All processing done locally
- No external data transmission
- Secure handling of sensitive information

### Resource Management

#### Memory Optimization
```yaml
# In docker-compose.yml
backend:
  deploy:
    resources:
      limits:
        memory: 8G
frontend:
  deploy:
    resources:
      limits:
        memory: 2G
```

#### CPU Optimization
- `RF_N_JOBS=4` to prevent OOM on Jetson
- `XGB_DEVICE="cpu"` for ARM64 compatibility
- Proper thread management

### Monitoring and Logging

#### Service Monitoring
```bash
# Check service logs
docker compose logs backend
docker compose logs frontend

# Follow real-time logs
docker compose logs -f backend
```

#### Health Checks
```bash
# Health check endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs  # API documentation
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failures
```bash
# Check credentials
grep ACLED .env

# Test authentication
python -c "
from src.data.acled_fetcher import get_acled_token
print('Token:', get_acled_token())
"
```

#### 2. Docker Build Failures
```bash
# Clear Docker cache
docker system prune -f

# Rebuild with no cache
docker compose build --no-cache
```

#### 3. Memory Issues
```bash
# Check memory usage
free -h

# Verify RF_N_JOBS setting
python -c "
from src.config import RF_N_JOBS
print('RF_N_JOBS:', RF_N_JOBS)
"
```

#### 4. Ollama Connectivity
```bash
# Test Ollama connection
docker compose exec backend curl -s http://host.docker.internal:11434/api/tags

# Ensure Ollama is running on host
ollama list
```

### Recovery Procedures

#### Restart Services
```bash
# Stop all services
docker compose down

# Start all services
docker compose up -d
```

#### Data Recovery
```bash
# Check data integrity
ls data/raw/
ls data/processed/

# Re-run data pipeline if needed
python scripts/fetch_data.py --synthetic
```

## Performance Optimization

### System Tuning

#### Docker Configuration
```yaml
# In docker-compose.yml
backend:
  environment:
    - RF_N_JOBS=4
    - XGB_DEVICE=cpu
  deploy:
    resources:
      limits:
        memory: 8G
        cpus: "4.0"
```

#### Memory Management
- Monitor memory usage with `htop`
- Set appropriate Docker memory limits
- Use swap space if needed
- Optimize Python memory usage

### Data Pipeline Optimization

#### Caching Strategy
- Cache API responses when possible
- Use efficient data structures
- Implement proper indexing
- Optimize database queries

#### Parallel Processing
- Use appropriate `n_jobs` parameter
- Implement batch processing
- Optimize I/O operations
- Use efficient algorithms

## Maintenance

### Regular Updates

#### Code Updates
```bash
# Pull latest code
git pull origin main

# Rebuild services
docker compose build
docker compose up -d
```

#### Model Updates
```bash
# Update Ollama model
ollama pull qwen3:32b

# Restart services
docker compose restart backend
```

### Backup Procedures

#### Data Backup
```bash
# Backup processed data
tar -czf backup-$(date +%Y%m%d).tar.gz data/processed/

# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/
```

#### Configuration Backup
```bash
# Backup environment
cp .env .env.backup-$(date +%Y%m%d)
```

## Scaling Considerations

### Horizontal Scaling
- Backend services can be scaled horizontally
- Load balancing between instances
- Shared data storage
- Database connection pooling

### Vertical Scaling
- Increase memory limits in Docker
- Add more CPU cores
- Expand storage capacity
- Optimize database performance

### Cloud Deployment
- AWS EC2 Graviton instances
- Azure ARM64 VMs
- Google Cloud ARM64 VMs
- Container orchestration with Kubernetes

## Monitoring and Alerting

### System Metrics
- CPU utilization
- Memory usage
- Disk space
- Network I/O
- API response times

### Log Analysis
- Error logging
- Performance metrics
- Data quality checks
- Authentication logs

### Alerting Setup
- Email notifications
- Slack integration
- SMS alerts
- Dashboard monitoring

## Compliance and Legal

### Data Privacy
- GDPR compliance
- Data retention policies
- Access controls
- Audit trails

### Security Compliance
- Regular security scans
- Vulnerability assessments
- Penetration testing
- Compliance reporting

## Support and Documentation

### Documentation Updates
- Keep documentation current
- Update API documentation
- Add usage examples
- Maintain troubleshooting guides

### Support Channels
- GitHub issue tracker
- Community forums
- Email support
- Documentation site

## Conclusion

The FamineSight system is designed for robust, production-ready deployment on ARM64 platforms including Jetson AGX Orin. With proper configuration and monitoring, the system provides reliable humanitarian data analysis capabilities for famine prediction and response planning.

Regular maintenance, security updates, and performance monitoring are essential for sustained operation in production environments.