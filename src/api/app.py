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
from .pricing_models import (
    OwnerRecommendRequest, OwnerRecommendResponse,
    RenterAlertRequest, RenterAlertResponse,
    InvestorOpportunityRequest, InvestorOpportunityResponse,
    CurrentPriceRequest, CurrentPriceResponse,
)
from .utils import load_model, prepare_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
monitor = None
pricing_engine = None

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

# Initialize pricing engine
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pricing import PricingEngine
    from pricing.constraints import PriceConstraints, MarketPosition
    pricing_engine = PricingEngine()
    logger.info("Pricing engine initialized")
except Exception as e:
    logger.warning(f"Pricing engine not available: {e}")
    pricing_engine = None


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
        "health": "/health",
        "pricing": "/pricing/recommend, /pricing/alert, /pricing/opportunity, /pricing/current"
    }


def _extract_location(property_data: dict) -> str:
    """Extract location string from one-hot encoded location fields."""
    if property_data.get("location_Downtown", 0) == 1:
        return "Downtown"
    if property_data.get("location_Urban", 0) == 1:
        return "Urban"
    if property_data.get("location_Suburbs", 0) == 1:
        return "Suburbs"
    if property_data.get("location_Rural", 0) == 1:
        return "Rural"
    return "Suburbs"  # Default fallback


def _get_base_price(property_data: dict) -> float:
    """Get base price prediction from ML model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features_df = prepare_features(property_data)
    prediction = model.predict(features_df)[0]
    return float(prediction)


@app.post("/pricing/recommend", response_model=OwnerRecommendResponse)
async def recommend_price(request: OwnerRecommendRequest):
    """
    Recommend list price for property owner based on market conditions.
    """
    if pricing_engine is None:
        raise HTTPException(status_code=503, detail="Pricing engine not available")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        property_dict = request.property.model_dump()
        location = _extract_location(property_dict)
        base_price = _get_base_price(property_dict)
        
        # Parse market position
        mp_str = request.market_position.lower()
        if mp_str == "aggressive":
            market_pos = MarketPosition.AGGRESSIVE
        elif mp_str == "conservative":
            market_pos = MarketPosition.CONSERVATIVE
        else:
            market_pos = MarketPosition.MARKET
        
        constraints = PriceConstraints(
            min_price=request.min_price,
            max_price=request.max_price,
            market_position=market_pos,
        )
        
        as_of_date = None
        if request.as_of_date:
            from datetime import datetime
            as_of_date = datetime.strptime(request.as_of_date, "%Y-%m-%d").date()
        
        rec = pricing_engine.recommend_for_owner(
            base_price=base_price,
            location=location,
            constraints=constraints,
            as_of_date=as_of_date,
        )
        
        return OwnerRecommendResponse(
            recommended_price=rec.recommended_price,
            price_min=rec.price_min,
            price_max=rec.price_max,
            base_price=base_price,
            demand_level=rec.demand_level,
            market_position_used=rec.market_position_used,
            reasoning=rec.reasoning,
        )
    except Exception as e:
        logger.error(f"Pricing recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Pricing recommendation failed: {str(e)}")


@app.post("/pricing/alert", response_model=RenterAlertResponse)
async def price_alert(request: RenterAlertRequest):
    """
    Check if asking price is fair for a renter based on market conditions.
    """
    if pricing_engine is None:
        raise HTTPException(status_code=503, detail="Pricing engine not available")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        property_dict = request.property.model_dump()
        location = _extract_location(property_dict)
        base_price = _get_base_price(property_dict)
        
        as_of_date = None
        if request.as_of_date:
            from datetime import datetime
            as_of_date = datetime.strptime(request.as_of_date, "%Y-%m-%d").date()
        
        alert = pricing_engine.alert_for_renter(
            asking_price=request.asking_price,
            base_price=base_price,
            location=location,
            as_of_date=as_of_date,
            fair_band_pct=request.fair_band_pct,
        )
        
        return RenterAlertResponse(
            is_fair=alert.is_fair,
            asking_price=alert.asking_price,
            fair_low=alert.fair_low,
            fair_high=alert.fair_high,
            base_price=base_price,
            message=alert.message,
        )
    except Exception as e:
        logger.error(f"Price alert error: {e}")
        raise HTTPException(status_code=500, detail=f"Price alert failed: {str(e)}")


@app.post("/pricing/opportunity", response_model=InvestorOpportunityResponse)
async def investor_opportunity(request: InvestorOpportunityRequest):
    """
    Score investment opportunity: suggested bid, expected value, ROI check.
    """
    if pricing_engine is None:
        raise HTTPException(status_code=503, detail="Pricing engine not available")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        property_dict = request.property.model_dump()
        location = _extract_location(property_dict)
        base_price = _get_base_price(property_dict)
        
        as_of_date = None
        if request.as_of_date:
            from datetime import datetime
            as_of_date = datetime.strptime(request.as_of_date, "%Y-%m-%d").date()
        
        opp = pricing_engine.opportunity_for_investor(
            base_price=base_price,
            location=location,
            as_of_date=as_of_date,
            min_roi_pct=request.min_roi_pct,
            list_discount_pct=request.list_discount_pct,
        )
        
        return InvestorOpportunityResponse(
            score=opp.score,
            suggested_bid=opp.suggested_bid,
            expected_value=opp.expected_value,
            base_price=base_price,
            meets_roi=opp.meets_roi,
            reasoning=opp.reasoning,
        )
    except Exception as e:
        logger.error(f"Investor opportunity error: {e}")
        raise HTTPException(status_code=500, detail=f"Investor opportunity failed: {str(e)}")


@app.post("/pricing/current", response_model=CurrentPriceResponse)
async def current_price(request: CurrentPriceRequest):
    """
    Get current dynamic (surge-style) price based on real-time market conditions.
    """
    if pricing_engine is None:
        raise HTTPException(status_code=503, detail="Pricing engine not available")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        property_dict = request.property.model_dump()
        location = _extract_location(property_dict)
        base_price = _get_base_price(property_dict)
        
        as_of_date = None
        if request.as_of_date:
            from datetime import datetime
            as_of_date = datetime.strptime(request.as_of_date, "%Y-%m-%d").date()
        
        current = pricing_engine.current_dynamic_price(
            base_price=base_price,
            location=location,
            as_of_date=as_of_date,
        )
        
        return CurrentPriceResponse(
            current_price=current["current_price"],
            base_price=base_price,
            demand_multiplier=current["demand_multiplier"],
            demand_level=current["demand_level"],
            competition_effect=current["competition_effect"],
        )
    except Exception as e:
        logger.error(f"Current price error: {e}")
        raise HTTPException(status_code=500, detail=f"Current price failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
