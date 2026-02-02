# PropTech MLOps - Model Drift Detection System

A learning project demonstrating end-to-end MLOps practices for property price prediction with drift detection and automated retraining.

## Project Overview

This project builds a production-like ML system that:

- Predicts residential property prices using ML models
- Monitors for data and prediction drift
- Automatically retrains models when drift is detected
- Uses free, open-source tools throughout

## Phase 1: Foundation & Data Generation

### Setup

1. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate baseline dataset:

```bash
python src/data/generate_data.py
```

This creates `data/raw/baseline_properties.csv` with synthetic property listings.

## Project Structure

```
prop-ops/
├── src/
│   ├── data/           # Data generation & processing
│   ├── training/       # Model training scripts
│   ├── api/            # FastAPI inference service
│   ├── monitoring/     # Drift detection & monitoring
│   ├── dashboard/      # Visualization dashboard
│   └── pipelines/      # Airflow DAGs & retraining logic
├── data/
│   ├── raw/            # Raw datasets
│   └── processed/      # Processed datasets
└── requirements.txt
```

## Next Steps

- Phase 2: Data versioning with DVC
- Phase 3: Model training with MLflow
- Phase 4: FastAPI inference service
- Phase 5: Monitoring & drift detection
- Phase 6: Dashboard visualization
- Phase 7: Automated retraining pipeline
- Phase 8: Docker Compose orchestration
