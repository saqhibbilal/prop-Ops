"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PropertyInput(BaseModel):
    """Single property input for prediction."""
    area_sqft: float = Field(..., description="Property area in square feet", gt=0)
    bedrooms: int = Field(..., description="Number of bedrooms", ge=0)
    bathrooms: float = Field(..., description="Number of bathrooms", ge=0)
    age: int = Field(..., description="Property age (years)", ge=0)
    has_parking: int = Field(..., description="Has parking (0 or 1)", ge=0, le=1)
    has_gym: int = Field(..., description="Has gym (0 or 1)", ge=0, le=1)
    has_pool: int = Field(..., description="Has pool (0 or 1)", ge=0, le=1)
    price_per_sqft: Optional[float] = Field(None, description="Price per sqft (calculated if not provided)")
    total_rooms: Optional[float] = Field(None, description="Total rooms (calculated if not provided)")
    property_type_Apartment: int = Field(0, description="Property type: Apartment", ge=0, le=1)
    property_type_Condo: int = Field(0, description="Property type: Condo", ge=0, le=1)
    property_type_House: int = Field(0, description="Property type: House", ge=0, le=1)
    property_type_Townhouse: int = Field(0, description="Property type: Townhouse", ge=0, le=1)
    location_Downtown: int = Field(0, description="Location: Downtown", ge=0, le=1)
    location_Rural: int = Field(0, description="Location: Rural", ge=0, le=1)
    location_Suburbs: int = Field(0, description="Location: Suburbs", ge=0, le=1)
    location_Urban: int = Field(0, description="Location: Urban", ge=0, le=1)


class PropertyPrediction(BaseModel):
    """Prediction response."""
    price: float = Field(..., description="Predicted property price")
    price_formatted: str = Field(..., description="Formatted price string")


class BatchPropertyInput(BaseModel):
    """Batch property inputs for prediction."""
    properties: List[PropertyInput] = Field(..., description="List of properties to predict")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PropertyPrediction] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
