"""
FastAPI inference service for property price prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
from contextlib import asynccontextmanager

from .models import (
    PropertyInput, PropertyPrediction, 
    BatchPropertyInput, BatchPredictionResponse,
    HealthResponse
)
from .utils import load_model, prepare_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
monitor = None

# Initialize monitoring
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from monitoring.monitor import PredictionMonitor
    monitor = PredictionMonitor()
    logger.info("Monitoring initialized")
except Exception as e:
    logger.warning(f"Monitoring not available: {e}")
    monitor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    try:
        model_uri = os.getenv("MLFLOW_MODEL_URI")
        run_id = os.getenv("MLFLOW_RUN_ID")
        model = load_model(model_uri=model_uri, run_id=run_id)
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        model = None
    yield
    # Cleanup on shutdown
    model = None


# Create FastAPI app
app = FastAPI(
    title="Property Price Prediction API",
    description="ML inference service for predicting residential property prices",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PropertyPrediction)
async def predict_single(property_input: PropertyInput):
    """
    Predict price for a single property.
    
    Args:
        property_input: Property features
    
    Returns:
        Predicted price
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict and prepare features
        property_dict = property_input.model_dump()
        features_df = prepare_features(property_dict)
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Log prediction for monitoring
        if monitor:
            try:
                monitor.log_prediction(
                    features=property_dict,
                    prediction=float(prediction),
                    model_version=os.getenv("MLFLOW_RUN_ID")
                )
            except Exception as e:
                logger.warning(f"Failed to log prediction: {e}")
        
        return PropertyPrediction(
            price=float(prediction),
            price_formatted=f"${prediction:,.2f}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchPropertyInput):
    """
    Predict prices for multiple properties.
    
    Args:
        batch_input: List of property features
    
    Returns:
        List of predicted prices
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for prop in batch_input.properties:
            property_dict = prop.model_dump()
            features_df = prepare_features(property_dict)
            prediction = model.predict(features_df)[0]
            
            pred_obj = PropertyPrediction(
                price=float(prediction),
                price_formatted=f"${prediction:,.2f}"
            )
            predictions.append(pred_obj)
            
            # Log prediction for monitoring
            if monitor:
                try:
                    monitor.log_prediction(
                        features=property_dict,
                        prediction=float(prediction),
                        model_version=os.getenv("MLFLOW_RUN_ID")
                    )
                except Exception as e:
                    logger.warning(f"Failed to log prediction: {e}")
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Property Price Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
