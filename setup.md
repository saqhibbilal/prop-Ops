# Setup guide – run the project after cloning

Follow these steps in order. Everything runs on your machine; no cloud needed.

---

## 1. Setup (once per clone)

Open a terminal in the project folder.

**Create and activate a virtual environment:**

- **Windows (PowerShell):**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

You only need to do this once (or when `requirements.txt` changes).

---

## 2. Generate data

The app needs property data and processed train/val/test files.

**Generate baseline property data:**

```bash
python src/data/generate_data.py
```

**Build train/val/test splits and features:**

```bash
python src/data/process_data.py data/raw/baseline_properties.csv data/processed
```

After this you should see files under `data/processed/` (e.g. `train.csv`, `val.csv`, `test.csv`).

---

## 3. Train models

**Train the main prediction model (XGBoost):**

```bash
python -m src.training.train
```

**Train all comparison models (LightGBM, Random Forest, Linear Regression, Ridge, Lasso):**

```bash
python -m src.training.train_multiple_models
```

**Optional – run hyperparameter tuning (XGBoost + LightGBM):**

```bash
python -m src.training.hyperparameter_tuning --model both
```

Training logs go to MLflow (under `mlruns/`). You can start the MLflow UI later with `mlflow ui` if you want to browse experiments.

---

## 4. Generate market signals (for dynamic pricing)

The dashboard and API use pre-generated demand/competition signals.

**Generate and store market signals (about 2 years of data):**

```bash
python -m src.data.generate_market_signals
```

This creates (or updates) `data/market_signals.db`. Run it once; the dashboard and pricing endpoints need it.

---

## 5. Run the app

You can run the API and the dashboard; they work together but the dashboard can also use the pricing engine directly (without the API) for the Dynamic Pricing tab.

**Start the API** (in one terminal):

```bash
uvicorn src.api.app:app --reload --port 8000
```

- API base URL: **http://localhost:8000**
- Docs: **http://localhost:8000/docs**

**Start the dashboard** (in a second terminal, with the same venv activated):

```bash
streamlit run src/dashboard/app.py
```

- Dashboard: **http://localhost:8501**

Use the dashboard to see predictions, drift, model comparison, tuning results, and dynamic pricing (owner/renter/investor tools and market signals).

---

## Project layout (where things live)

```
prop-ops/
├── src/
│   ├── data/           → generate_data, process_data, market signals
│   ├── training/       → train, train_multiple_models, tuning
│   ├── api/            → FastAPI (predict + pricing endpoints)
│   ├── pricing/        → dynamic pricing logic
│   ├── monitoring/     → drift detection
│   ├── dashboard/      → Streamlit UI
│   └── pipelines/      → retraining (e.g. for Airflow)
├── data/
│   ├── raw/            → raw data (DVC-tracked)
│   ├── processed/      → train/val/test CSVs
│   └── market_signals.db
├── mlruns/             → MLflow runs (created when you train)
├── requirements.txt
└── docker-compose.yml  → optional: run everything in Docker
```

---

## API endpoints (when the API is running)

**Predictions:**

- `POST /predict` – single property price
- `POST /predict/batch` – batch predictions
- `GET /health` – health check

**Dynamic pricing:**

- `POST /pricing/recommend` – owner recommendation
- `POST /pricing/alert` – renter fairness check
- `POST /pricing/opportunity` – investor opportunity score
- `POST /pricing/current` – current dynamic (surge-style) price

Example (owner recommendation):

```bash
curl -X POST http://localhost:8000/pricing/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "property": {
      "area_sqft": 1500,
      "bedrooms": 3,
      "bathrooms": 2,
      "age": 10,
      "has_parking": 1,
      "has_gym": 1,
      "has_pool": 0,
      "property_type_House": 1,
      "location_Suburbs": 1
    },
    "market_position": "market"
  }'
```

---

## Dashboard tabs

1. **Overview** – metrics and recent predictions
2. **Predictions** – prediction distribution and stats
3. **Drift Detection** – data and prediction drift
4. **Metrics** – model performance
5. **Model Comparison** – compare trained models
6. **Hyperparameter Tuning** – run tuning and see results
7. **Dynamic Pricing** – owner/renter/investor tools and market signals

---

## Testing

```bash
python test_api.py
python test_pricing.py    # needs API running on port 8000
python test_monitoring.py
```

---

## Environment variables (optional)

You can leave these unset; the app will use defaults (e.g. latest MLflow run for the model).

```bash
MLFLOW_TRACKING_URI=file:///path/to/mlruns
MLFLOW_MODEL_URI=runs:/<run_id>/model
MLFLOW_RUN_ID=<run_id>
MLFLOW_EXPERIMENT_NAME=property_price_prediction
REFERENCE_DATA_PATH=data/processed/train.csv
```

---

## Dependencies

See `requirements.txt`. Main ones: pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, mlflow, fastapi, uvicorn, pydantic, evidently, streamlit, plotly, dvc. Optional: apache-airflow.

---

## Docker (optional)

To run everything in containers, see **DOCKER.md**.

```bash
docker-compose up -d
# Or: ./start_docker.sh  (Mac/Linux)  or  start_docker.ps1  (Windows)
```
