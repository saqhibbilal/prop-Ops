# Dockerfiles

This directory contains Dockerfiles for different services.

## Files

- `Dockerfile.api` - FastAPI inference service
- `Dockerfile.dashboard` - Streamlit monitoring dashboard
- `Dockerfile.mlflow` - MLflow tracking server

## Building

Build all images:

```bash
docker-compose build
```

Build individual service:

```bash
docker build -f docker/Dockerfile.api -t prop-ops-api .
```

## Base Image

All services use `python:3.9-slim` for smaller image size.

## Dependencies

All dependencies are installed from `requirements.txt` in the project root.
