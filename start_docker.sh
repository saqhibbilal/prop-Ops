#!/bin/bash
# Startup script for Docker Compose

echo "Starting PropTech MLOps Services..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Start services
echo "Building and starting Docker containers..."
docker-compose up --build -d

echo ""
echo "Services starting..."
echo ""
echo "MLflow UI:    http://localhost:5000"
echo "API:          http://localhost:8000"
echo "API Docs:     http://localhost:8000/docs"
echo "Dashboard:    http://localhost:8501"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop:      docker-compose down"
