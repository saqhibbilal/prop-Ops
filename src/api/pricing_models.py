"""
Pydantic models for dynamic pricing API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class PricingPropertyInput(BaseModel):
    """Property input for pricing (reuses PropertyInput structure)."""
    area_sqft: float = Field(..., description="Property area in square feet", gt=0)
    bedrooms: int = Field(..., description="Number of bedrooms", ge=0)
    bathrooms: float = Field(..., description="Number of bathrooms", ge=0)
    age: int = Field(..., description="Property age (years)", ge=0)
    has_parking: int = Field(..., description="Has parking (0 or 1)", ge=0, le=1)
    has_gym: int = Field(..., description="Has gym (0 or 1)", ge=0, le=1)
    has_pool: int = Field(..., description="Has pool (0 or 1)", ge=0, le=1)
    property_type_Apartment: int = Field(0, description="Property type: Apartment", ge=0, le=1)
    property_type_Condo: int = Field(0, description="Property type: Condo", ge=0, le=1)
    property_type_House: int = Field(0, description="Property type: House", ge=0, le=1)
    property_type_Townhouse: int = Field(0, description="Property type: Townhouse", ge=0, le=1)
    location_Downtown: int = Field(0, description="Location: Downtown", ge=0, le=1)
    location_Rural: int = Field(0, description="Location: Rural", ge=0, le=1)
    location_Suburbs: int = Field(0, description="Location: Suburbs", ge=0, le=1)
    location_Urban: int = Field(0, description="Location: Urban", ge=0, le=1)


class OwnerRecommendRequest(BaseModel):
    """Request for owner pricing recommendation."""
    property: PricingPropertyInput = Field(..., description="Property features")
    market_position: Optional[str] = Field("market", description="Market position: aggressive, market, conservative")
    min_price: Optional[float] = Field(None, description="Minimum price constraint")
    max_price: Optional[float] = Field(None, description="Maximum price constraint")
    as_of_date: Optional[str] = Field(None, description="Date for market conditions (YYYY-MM-DD)")


class OwnerRecommendResponse(BaseModel):
    """Owner pricing recommendation response."""
    recommended_price: float = Field(..., description="Recommended list price")
    price_min: float = Field(..., description="Minimum recommended price")
    price_max: float = Field(..., description="Maximum recommended price")
    base_price: float = Field(..., description="Base ML model prediction")
    demand_level: str = Field(..., description="Current demand level")
    market_position_used: str = Field(..., description="Market position applied")
    reasoning: str = Field(..., description="Explanation of recommendation")


class RenterAlertRequest(BaseModel):
    """Request for renter price alert."""
    property: PricingPropertyInput = Field(..., description="Property features")
    asking_price: float = Field(..., description="Asking/listed price", gt=0)
    fair_band_pct: Optional[float] = Field(0.08, description="Fair price band percentage", ge=0, le=0.2)
    as_of_date: Optional[str] = Field(None, description="Date for market conditions (YYYY-MM-DD)")


class RenterAlertResponse(BaseModel):
    """Renter price alert response."""
    is_fair: bool = Field(..., description="Whether asking price is fair")
    asking_price: float = Field(..., description="Asking price")
    fair_low: float = Field(..., description="Lower bound of fair price range")
    fair_high: float = Field(..., description="Upper bound of fair price range")
    base_price: float = Field(..., description="Base ML model prediction")
    message: str = Field(..., description="Alert message")


class InvestorOpportunityRequest(BaseModel):
    """Request for investor opportunity scoring."""
    property: PricingPropertyInput = Field(..., description="Property features")
    min_roi_pct: Optional[float] = Field(8.0, description="Minimum ROI percentage target", ge=0)
    list_discount_pct: Optional[float] = Field(0.05, description="Suggested discount from list", ge=0, le=0.2)
    as_of_date: Optional[str] = Field(None, description="Date for market conditions (YYYY-MM-DD)")


class InvestorOpportunityResponse(BaseModel):
    """Investor opportunity response."""
    score: float = Field(..., description="Opportunity score (0-100)")
    suggested_bid: float = Field(..., description="Suggested bid price")
    expected_value: float = Field(..., description="Expected market value")
    base_price: float = Field(..., description="Base ML model prediction")
    meets_roi: bool = Field(..., description="Whether ROI target is met")
    reasoning: str = Field(..., description="Explanation of opportunity")


class CurrentPriceRequest(BaseModel):
    """Request for current dynamic (surge-style) price."""
    property: PricingPropertyInput = Field(..., description="Property features")
    as_of_date: Optional[str] = Field(None, description="Date for market conditions (YYYY-MM-DD)")


class CurrentPriceResponse(BaseModel):
    """Current dynamic price response."""
    current_price: float = Field(..., description="Current dynamic price")
    base_price: float = Field(..., description="Base ML model prediction")
    demand_multiplier: float = Field(..., description="Demand multiplier applied")
    demand_level: str = Field(..., description="Current demand level")
    competition_effect: float = Field(..., description="Competition effect multiplier")
