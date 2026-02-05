# PropTech MLOps – Model Drift Detection & Dynamic Pricing

This is a **learning project** that demonstrates an end-to-end MLOps system for property price prediction, drift detection, and dynamic pricing using open-source tools.

**What it is.** The project builds a production-style ML system that predicts residential property prices, monitors for data and prediction drift, supports automated retraining, and exposes dynamic pricing features for owners, renters, and investors. Data is synthetic (property features and market signals), with pipelines and practices intended to mirror a real PropTech stack.

**MLOps pipeline.** The pipeline covers the full lifecycle: data versioning with DVC (baseline and processed datasets), feature engineering and train/val/test splits, training with multiple algorithms (XGBoost, LightGBM, Random Forest, Linear Regression, Ridge, Lasso), experiment tracking and model registry in MLflow, hyperparameter tuning with Optuna, model serving via FastAPI, drift detection with Evidently AI, and a Streamlit dashboard for metrics, model comparison, tuning, and pricing. Retraining can be triggered manually or via an Airflow DAG when drift is detected.

**Tech stack.** Core stack includes Python, pandas/numpy for data; scikit-learn, XGBoost, LightGBM for models; MLflow for tracking and registry; DVC for data versioning; FastAPI for the API; Streamlit and Plotly for the dashboard; Evidently for drift; Optuna for tuning; and optional Docker Compose and Airflow for deployment and scheduling. All components are free and open source.

**Dynamic pricing.** On top of the predicted base price, the system adds a dynamic pricing layer: owner recommendations (market position and constraints), renter fairness checks (asking price vs fair range), investor opportunity scoring (ROI and suggested bid), and a “current” surge-style price driven by demand and competition. Market signals (demand multiplier, competition listings, seasonality) are pre-generated and stored in SQLite, then used by the pricing engine and visualized in the dashboard.

**Future enhancements.** Next steps include plugging in real property and market data (listings, transactions, demand proxies), hardening the API (auth, rate limits), adding alerts (e.g. Slack/email on drift or model promotion), and optionally running retraining on a schedule or on drift. The current design is ready for those extensions.

---

## Screenshots

| Dashboard overview                                          | MLflow – experiment metrics (MAE, R², RMSE)             |
| ----------------------------------------------------------- | ------------------------------------------------------- |
| ![Dashboard overview](Screenshot%202026-02-02%20193256.jpg) | ![MLflow metrics](Screenshot%202026-02-05%20062639.jpg) |

| Model comparison (predictions by model)                        | Hyperparameter tuning – best params                     |
| -------------------------------------------------------------- | ------------------------------------------------------- |
| ![Prediction comparison](Screenshot%202026-02-05%20062727.jpg) | ![Tuning results](Screenshot%202026-02-05%20062813.jpg) |

| Market signals – demand & competition over time         |
| ------------------------------------------------------- |
| ![Market signals](Screenshot%202026-02-05%20062915.jpg) |

---

## Features

### Core ML Pipeline

- Synthetic data generation with drift simulation
- Data versioning with DVC
- Model training with MLflow experiment tracking
- Multiple model comparison (5 algorithms)
- Hyperparameter tuning with Optuna
- Model serving via FastAPI
- Drift detection and monitoring
- Automated retraining pipeline

### Dynamic Pricing

- Owner pricing recommendations (market-aware)
- Renter price alerts (fairness checks)
- Investor opportunity scoring (ROI analysis)
- Current dynamic pricing (surge-style)
- Market signals visualization

**To run the project after cloning:** see **[setup.md](setup.md)** for step-by-step setup, data generation, training, and running the API and dashboard.

## License

This is a learning project. Use freely for educational purposes.
