# Docker Deployment Guide

This guide explains how to run the PropTech MLOps system using Docker Compose.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)

## Quick Start

1. **Build and start all services:**

```bash
docker-compose up --build
```

2. **Access services:**

   - **MLflow UI**: http://localhost:5000
   - **API**: http://localhost:8000
   - **Dashboard**: http://localhost:8501
   - **API Docs**: http://localhost:8000/docs

3. **Stop services:**

```bash
docker-compose down
```

## Services

### MLflow Tracking Server

- **Port**: 5000
- **Purpose**: Model experiment tracking and registry
- **Data**: Stored in `mlflow-data` volume

### FastAPI Inference Service

- **Port**: 8000
- **Purpose**: Serves model predictions via REST API
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions

### Streamlit Dashboard

- **Port**: 8501
- **Purpose**: Monitoring dashboard for predictions and drift
- **Features**: Overview, Predictions, Drift Detection, Metrics

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

Key variables:

- `MLFLOW_RUN_ID`: Run ID of the model to load (optional, will use latest if not set)
- `MLFLOW_TRACKING_URI`: MLflow server URI (default: http://mlflow:5000)

### Volumes

Data is persisted in Docker volumes:

- `mlflow-data`: MLflow experiments and models
- `monitoring-data`: Monitoring database and metrics

For development, local directories are also mounted:

- `./mlruns` → MLflow data
- `./monitoring` → Monitoring data

## Development Workflow

### 1. Train a Model First

Before starting services, train a model:

```bash
python src/training/train.py
```

Note the run_id from the output.

### 2. Set Model Run ID (Optional)

Edit `.env` file:

```
MLFLOW_RUN_ID=your_run_id_here
```

Or pass when starting:

```bash
MLFLOW_RUN_ID=your_run_id docker-compose up
```

### 3. Start Services

```bash
docker-compose up --build
```

### 4. Test API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "age": 10,
    "has_parking": 1,
    "has_gym": 1,
    "has_pool": 0,
    "property_type_House": 1,
    "location_Suburbs": 1
  }'
```

## Troubleshooting

### Services won't start

- Check if ports 5000, 8000, 8501 are available
- Check Docker logs: `docker-compose logs`

### API can't connect to MLflow

- Ensure MLflow service is healthy: `docker-compose ps`
- Check MLflow logs: `docker-compose logs mlflow`

### Model not loading

- Verify MLFLOW_RUN_ID is correct
- Check if model exists in MLflow UI: http://localhost:5000
- Check API logs: `docker-compose logs api`

### Dashboard shows no data

- Make some predictions via API first
- Check monitoring database exists
- Check dashboard logs: `docker-compose logs dashboard`

## Building Individual Services

### Build API only:

```bash
docker build -f docker/Dockerfile.api -t prop-ops-api .
```

### Build Dashboard only:

```bash
docker build -f docker/Dockerfile.dashboard -t prop-ops-dashboard .
```

### Build MLflow only:

```bash
docker build -f docker/Dockerfile.mlflow -t prop-ops-mlflow .
```

## Running Individual Services

### Run API standalone:

```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -v $(pwd)/src:/app/src \
  prop-ops-api
```

## Data Persistence

Data persists in Docker volumes even after stopping containers:

- To remove all data: `docker-compose down -v`
- To backup: `docker volume ls` then `docker volume inspect <volume_name>`

## Production Considerations

For production deployment:

1. Use environment-specific `.env` files
2. Set up proper secrets management
3. Use external database for MLflow (PostgreSQL)
4. Add reverse proxy (nginx) for HTTPS
5. Set up monitoring and logging (Prometheus, Grafana)
6. Use Kubernetes instead of Docker Compose for scaling

## Clean Up

Stop and remove containers:

```bash
docker-compose down
```

Remove containers, networks, and volumes:

```bash
docker-compose down -v
```

Remove images:

```bash
docker-compose down --rmi all
```
