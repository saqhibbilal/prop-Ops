"""
Streamlit dashboard for monitoring ML model predictions and drift detection.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.monitor import PredictionMonitor
from monitoring.config import MONITORING_DB

# Lazy import of DriftDetector to avoid asyncio issues
_detector = None

def get_detector():
    """Lazy load drift detector."""
    global _detector
    if _detector is None:
        try:
            from monitoring.drift_detector import DriftDetector
            _detector = DriftDetector()
        except Exception as e:
            st.warning(f"Drift detection not available: {e}")
            _detector = None
    return _detector

# Page configuration
st.set_page_config(
    page_title="PropTech ML Monitoring Dashboard",
    page_icon=None,
    layout="wide"
)

# Load custom CSS
_css_path = Path(__file__).parent / "assets" / "custom.css"
if _css_path.exists():
    with open(_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Chart color theme (Primary #2872A1, Secondary #CBDDE9)
CHART_COLORS = ["#2872A1", "#CBDDE9", "#5a9bc4", "#9ec9e0", "#1e5a82"]

# Initialize session state
if 'monitor' not in st.session_state:
    st.session_state.monitor = PredictionMonitor()

# Title
st.title("PropTech ML Monitoring Dashboard")
st.markdown("Monitor model predictions, detect drift, and track model performance")

# Sidebar
st.sidebar.header("Dashboard Controls")
refresh_interval = st.sidebar.slider("Auto-refresh interval (seconds)", 10, 300, 60)
auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)

if auto_refresh:
    st.experimental_rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Predictions",
    "Drift Detection",
    "Metrics",
    "Model Comparison"
])

# Tab 1: Overview
with tab1:
    st.header("System Overview")
    
    # Get recent data
    recent_predictions = st.session_state.monitor.get_recent_data(limit=1000)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = len(recent_predictions) if not recent_predictions.empty else 0
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        if not recent_predictions.empty:
            avg_prediction = recent_predictions['prediction'].mean()
            st.metric("Avg Prediction", f"${avg_prediction:,.0f}")
        else:
            st.metric("Avg Prediction", "N/A")
    
    with col3:
        if not recent_predictions.empty:
            unique_models = recent_predictions['model_version'].nunique()
            st.metric("Model Versions", unique_models)
        else:
            st.metric("Model Versions", "0")
    
    with col4:
        if not recent_predictions.empty:
            latest_time = pd.to_datetime(recent_predictions['timestamp']).max()
            st.metric("Last Prediction", latest_time.strftime("%H:%M:%S"))
        else:
            st.metric("Last Prediction", "N/A")
    
    # Recent predictions chart
    if not recent_predictions.empty:
        st.subheader("Recent Predictions Trend")
        recent_predictions['timestamp'] = pd.to_datetime(recent_predictions['timestamp'])
        recent_predictions_sorted = recent_predictions.sort_values('timestamp')
        
        fig = px.line(
            recent_predictions_sorted,
            x='timestamp',
            y='prediction',
            title='Prediction Trend Over Time',
            labels={'prediction': 'Predicted Price ($)', 'timestamp': 'Time'}
        )
        fig.update_traces(line_color=CHART_COLORS[0])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions data available yet. Make some predictions via the API to see data here.")

# Tab 2: Predictions
with tab2:
    st.header("Prediction Details")
    
    # Get predictions
    limit = st.slider("Number of predictions to display", 10, 500, 100)
    predictions_df = st.session_state.monitor.get_recent_data(limit=limit)
    
    if not predictions_df.empty:
        # Prediction distribution
        st.subheader("Prediction Distribution")
        fig = px.histogram(
            predictions_df,
            x='prediction',
            nbins=30,
            title='Distribution of Predictions',
            labels={'prediction': 'Predicted Price ($)', 'count': 'Frequency'}
        )
        fig.update_traces(marker_color=CHART_COLORS[0])
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistics")
            stats = predictions_df['prediction'].describe()
            st.dataframe(stats)
        
        with col2:
            st.subheader("Recent Predictions Table")
            display_df = predictions_df[['timestamp', 'prediction', 'model_version']].head(20)
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
            st.dataframe(display_df)
    else:
        st.info("No predictions data available.")

# Tab 3: Drift Detection
with tab3:
    st.header("Drift Detection")
    
    if st.button("Check for Drift"):
        detector = get_detector()
        if detector is None:
            st.error("Drift detection is not available. Evidently AI may not be properly installed or there's a compatibility issue.")
        else:
            with st.spinner("Analyzing data for drift..."):
                # Get recent data
                current_data = st.session_state.monitor.get_features_dataframe(limit=100)
                
                if not current_data.empty:
                    # Remove non-feature columns
                    feature_cols = [col for col in current_data.columns 
                                  if col not in ['prediction', 'timestamp', 'ground_truth']]
                    current_features = current_data[feature_cols] if feature_cols else pd.DataFrame()
                    
                    if not current_features.empty:
                        drift_results = detector.check_drift(current_data=current_features)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Overall Status")
                            drift_detected = drift_results.get('overall_drift_detected', False)
                            status = drift_results.get('status', 'unknown')
                            
                            if drift_detected:
                                st.error(f"Drift Detected. Status: {status}")
                            else:
                                st.success(f"No Drift Detected. Status: {status}")
                        
                        with col2:
                            st.subheader("Data Drift")
                            data_drift = drift_results.get('data_drift', {})
                            if data_drift:
                                drift_score = data_drift.get('drift_score', 0.0)
                                threshold = data_drift.get('threshold', 0.3)
                                st.metric("Drift Score", f"{drift_score:.4f}", delta=f"Threshold: {threshold}")
                        
                        # Drift metrics history
                        st.subheader("Drift Metrics History")
                        drift_metrics = st.session_state.monitor.store.get_drift_metrics(limit=50)
                        
                        if not drift_metrics.empty:
                            drift_metrics['timestamp'] = pd.to_datetime(drift_metrics['timestamp'])
                            
                            fig = px.line(
                                drift_metrics.sort_values('timestamp'),
                                x='timestamp',
                                y='metric_value',
                                color='metric_name',
                                title='Drift Metrics Over Time',
                                labels={'metric_value': 'Drift Score', 'timestamp': 'Time'}
                            )
                            fig.update_layout(colorway=CHART_COLORS)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(drift_metrics[['timestamp', 'metric_type', 'metric_name', 'metric_value', 'status']])
                        else:
                            st.info("No drift metrics history available yet.")
                    else:
                        st.warning("Not enough feature data for drift detection.")
                else:
                    st.warning("No recent data available for drift detection. Make some predictions first.")
    else:
        st.info("Click the button above to check for drift in the current data.")

# Tab 4: Metrics
with tab4:
    st.header("Model Metrics")
    
    predictions_df = st.session_state.monitor.get_recent_data(limit=1000)
    
    if not predictions_df.empty:
        # Feature analysis
        st.subheader("Feature Analysis")
        
        # Get features dataframe
        features_df = st.session_state.monitor.get_features_dataframe(limit=100)
        
        if not features_df.empty:
            # Select feature to analyze
            numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_features = [f for f in numeric_features if f not in ['prediction', 'ground_truth']]
            
            if numeric_features:
                selected_feature = st.selectbox("Select feature to analyze", numeric_features)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        features_df,
                        x=selected_feature,
                        title=f'Distribution of {selected_feature}',
                        nbins=20
                    )
                    fig.update_traces(marker_color=CHART_COLORS[0])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        features_df,
                        x=selected_feature,
                        y='prediction',
                        title=f'{selected_feature} vs Prediction',
                        labels={'prediction': 'Predicted Price ($)'}
                    )
                    fig.update_traces(marker_color=CHART_COLORS[0])
                    st.plotly_chart(fig, use_container_width=True)
        
        # Prediction vs Ground Truth (if available)
        if 'ground_truth' in predictions_df.columns and predictions_df['ground_truth'].notna().any():
            st.subheader("Prediction Accuracy")
            
            valid_data = predictions_df[predictions_df['ground_truth'].notna()]
            
            if not valid_data.empty:
                fig = px.scatter(
                    valid_data,
                    x='ground_truth',
                    y='prediction',
                    title='Predicted vs Actual Prices',
                    labels={'prediction': 'Predicted Price ($)', 'ground_truth': 'Actual Price ($)'}
                )
                fig.update_traces(marker_color=CHART_COLORS[0])
                max_val = max(valid_data['ground_truth'].max(), valid_data['prediction'].max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color=CHART_COLORS[4])
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                import numpy as np
                
                mae = mean_absolute_error(valid_data['ground_truth'], valid_data['prediction'])
                rmse = np.sqrt(mean_squared_error(valid_data['ground_truth'], valid_data['prediction']))
                r2 = r2_score(valid_data['ground_truth'], valid_data['prediction'])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"${mae:,.2f}")
                col2.metric("RMSE", f"${rmse:,.2f}")
                col3.metric("R² Score", f"{r2:.4f}")
    else:
        st.info("No metrics data available yet.")

# Tab 5: Model Comparison
with tab5:
    st.header("Model Comparison")
    try:
        from training.model_comparison import get_metrics_table, get_predictions_comparison
        metrics_df = get_metrics_table(use_test_data=True)
        if metrics_df.empty:
            st.info("No comparison runs yet. Run: `python -m src.training.train_multiple_models` from project root (or `python src/training/train_multiple_models.py`).")
        else:
            st.subheader("Metrics comparison")
            display_cols = [c for c in metrics_df.columns if c != "run_id"]
            num_cols = [c for c in display_cols if c != "model_type" and metrics_df[c].dtype in ["float64", "int64"]]
            fmt_df = metrics_df[display_cols].copy()
            if num_cols:
                for c in num_cols:
                    fmt_df[c] = fmt_df[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            st.dataframe(fmt_df)
            if "run_id" in metrics_df.columns:
                st.caption("Run IDs: " + ", ".join(metrics_df["run_id"].astype(str).tolist()))

            st.subheader("Prediction comparison (same test sample)")
            X_sample, y_sample, pred_df = get_predictions_comparison(sample_size=200)
            if not pred_df.empty:
                selected_model = st.selectbox("Select model to highlight", options=["All"] + pred_df["model_type"].unique().tolist(), key="model_compare_select")
                fig = px.box(
                    pred_df,
                    x="model_type",
                    y="prediction",
                    title="Prediction distribution by model",
                    labels={"prediction": "Predicted Price ($)", "model_type": "Model"},
                )
                fig.update_traces(marker_color=CHART_COLORS[0])
                st.plotly_chart(fig, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    mean_by_model = pred_df.groupby("model_type")["prediction"].agg(["mean", "std"]).reset_index()
                    mean_by_model.columns = ["Model", "Mean prediction ($)", "Std"]
                    mean_by_model["Mean prediction ($)"] = mean_by_model["Mean prediction ($)"].apply(lambda x: f"${x:,.0f}")
                    mean_by_model["Std"] = mean_by_model["Std"].apply(lambda x: f"{x:.2f}")
                    st.dataframe(mean_by_model)
                with col2:
                    st.caption("Sample size: {} rows from test set.".format(len(y_sample)))
            else:
                st.info("Could not load predictions. Ensure test data and models are available.")
    except Exception as e:
        st.warning("Model comparison not available: " + str(e))

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Database:** `{MONITORING_DB}`")
st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
