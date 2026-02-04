"""
Pricing engine: owner recommendations, renter alerts, investor opportunity, surge-style current price.
Uses market analyzer and constraints.
"""

import sys
from pathlib import Path
from datetime import date
from typing import Optional, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from pricing.market_analyzer import MarketAnalyzer
from pricing.constraints import PriceConstraints, MarketPosition


@dataclass
class OwnerRecommendation:
    recommended_price: float
    price_min: float
    price_max: float
    demand_level: str
    market_position_used: str
    reasoning: str


@dataclass
class RenterAlert:
    is_fair: bool
    asking_price: float
    fair_low: float
    fair_high: float
    base_price: float
    message: str


@dataclass
class InvestorOpportunity:
    score: float  # 0-100
    suggested_bid: float
    expected_value: float
    meets_roi: bool
    reasoning: str


class PricingEngine:
    """
    Dynamic pricing: owner list price, renter fairness check, investor score, current (surge) price.
    """

    def __init__(self, market_analyzer: Optional[MarketAnalyzer] = None):
        self.analyzer = market_analyzer or MarketAnalyzer()

    def recommend_for_owner(
        self,
        base_price: float,
        location: str,
        constraints: Optional[PriceConstraints] = None,
        as_of_date: Optional[date] = None,
    ) -> OwnerRecommendation:
        """
        Recommend list price for an owner. Applies demand and competition from market,
        then market position (aggressive/market/conservative), then clamps to constraints.
        """
        constraints = constraints or PriceConstraints()
        cond = self.analyzer.get_conditions(as_of_date=as_of_date, location=location)
        demand_mult = cond["demand_multiplier"]
        comp_effect = self.analyzer.get_competition_effect(location, as_of_date)
        position_offset = constraints.position_offset()

        # Market-adjusted price: base * demand * competition * (1 + position)
        raw = base_price * demand_mult * comp_effect * (1.0 + position_offset)
        recommended = constraints.clamp(raw, base_price)

        # Fair range for display (e.g. ±5% around recommended)
        price_min = constraints.clamp(recommended * 0.95, base_price)
        price_max = constraints.clamp(recommended * 1.05, base_price)

        reasoning = (
            f"Demand {cond['demand_level']} (mult {demand_mult:.2f}), "
            f"competition effect {comp_effect:.2f}, position {constraints.market_position.value}."
        )
        return OwnerRecommendation(
            recommended_price=round(recommended, 2),
            price_min=round(price_min, 2),
            price_max=round(price_max, 2),
            demand_level=cond["demand_level"],
            market_position_used=constraints.market_position.value,
            reasoning=reasoning,
        )

    def alert_for_renter(
        self,
        asking_price: float,
        base_price: float,
        location: str,
        as_of_date: Optional[date] = None,
        fair_band_pct: float = 0.08,
    ) -> RenterAlert:
        """
        Check if asking price is fair for a renter. Fair range = market-adjusted base ± band.
        """
        cond = self.analyzer.get_conditions(as_of_date=as_of_date, location=location)
        demand_mult = cond["demand_multiplier"]
        comp_effect = self.analyzer.get_competition_effect(location, as_of_date)
        fair_mid = base_price * demand_mult * comp_effect
        fair_low = fair_mid * (1.0 - fair_band_pct)
        fair_high = fair_mid * (1.0 + fair_band_pct)

        is_fair = fair_low <= asking_price <= fair_high
        if is_fair:
            message = f"Asking price is within fair range (${fair_low:,.0f} - ${fair_high:,.0f})."
        elif asking_price > fair_high:
            message = f"Asking price is above fair range (fair max ${fair_high:,.0f}). Consider negotiating."
        else:
            message = f"Asking price is below fair range (fair min ${fair_low:,.0f}). Good value."

        return RenterAlert(
            is_fair=is_fair,
            asking_price=asking_price,
            fair_low=round(fair_low, 2),
            fair_high=round(fair_high, 2),
            base_price=round(base_price, 2),
            message=message,
        )

    def opportunity_for_investor(
        self,
        base_price: float,
        location: str,
        as_of_date: Optional[date] = None,
        min_roi_pct: float = 8.0,
        list_discount_pct: float = 0.05,
    ) -> InvestorOpportunity:
        """
        Score investment opportunity: suggested bid below market, expected value, ROI check.
        Score 0-100: higher when below market and ROI target is met.
        """
        cond = self.analyzer.get_conditions(as_of_date=as_of_date, location=location)
        demand_mult = cond["demand_multiplier"]
        comp_effect = self.analyzer.get_competition_effect(location, as_of_date)
        expected_value = base_price * demand_mult * comp_effect
        suggested_bid = expected_value * (1.0 - list_discount_pct)

        # Simple ROI: (expected_value - bid) / bid * 100
        roi_pct = ((expected_value - suggested_bid) / suggested_bid * 100) if suggested_bid > 0 else 0.0
        meets_roi = roi_pct >= min_roi_pct

        # Score: base 50, +25 if below market, +25 if meets ROI
        score = 50.0
        if suggested_bid < expected_value:
            score += 25.0
        if meets_roi:
            score += 25.0
        score = min(100.0, score)

        reasoning = (
            f"Expected value ${expected_value:,.0f}; suggested bid ${suggested_bid:,.0f} ({list_discount_pct*100:.0f}% below). "
            f"ROI {roi_pct:.1f}% vs target {min_roi_pct}%."
        )
        return InvestorOpportunity(
            score=round(score, 1),
            suggested_bid=round(suggested_bid, 2),
            expected_value=round(expected_value, 2),
            meets_roi=meets_roi,
            reasoning=reasoning,
        )

    def current_dynamic_price(
        self,
        base_price: float,
        location: str,
        as_of_date: Optional[date] = None,
        min_pct: float = 0.85,
        max_pct: float = 1.25,
    ) -> Dict[str, Any]:
        """
        Surge-style current price: base * demand multiplier, clamped.
        Mimics real-time or day-of pricing (e.g. high demand = higher price).
        """
        cond = self.analyzer.get_conditions(as_of_date=as_of_date, location=location)
        demand_mult = cond["demand_multiplier"]
        comp_effect = self.analyzer.get_competition_effect(location, as_of_date)
        raw = base_price * demand_mult * comp_effect
        price = max(base_price * min_pct, min(base_price * max_pct, raw))
        return {
            "current_price": round(price, 2),
            "base_price": base_price,
            "demand_multiplier": demand_mult,
            "demand_level": cond["demand_level"],
            "competition_effect": comp_effect,
        }
