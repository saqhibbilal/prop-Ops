# PowerShell startup script for Docker Compose

Write-Host "Starting PropTech MLOps Services..." -ForegroundColor Green
Write-Host ""

# Check if .env exists
if (-not (Test-Path .env)) {
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "Please edit .env file with your configuration" -ForegroundColor Yellow
}

# Start services
Write-Host "Building and starting Docker containers..." -ForegroundColor Cyan
docker-compose up --build -d

Write-Host ""
Write-Host "Services starting..." -ForegroundColor Green
Write-Host ""
Write-Host "MLflow UI:    http://localhost:5000" -ForegroundColor Cyan
Write-Host "API:          http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs:     http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Dashboard:    http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view logs: docker-compose logs -f" -ForegroundColor Yellow
Write-Host "To stop:      docker-compose down" -ForegroundColor Yellow
