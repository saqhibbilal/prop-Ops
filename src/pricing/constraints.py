"""
Pricing constraints: price bounds, ROI targets, market positioning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarketPosition(str, Enum):
    """How to position price relative to market."""
    AGGRESSIVE = "aggressive"   # List above market (e.g. +3%)
    MARKET = "market"           # At market (0%)
    CONSERVATIVE = "conservative"  # List below market for faster sale (e.g. -2%)


@dataclass
class PriceConstraints:
    """
    Constraints for recommended or acceptable prices.
    All monetary values in same currency (e.g. USD).
    """
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_price_pct_of_base: Optional[float] = None   # e.g. 0.90 = no lower than 90% of base
    max_price_pct_of_base: Optional[float] = None   # e.g. 1.15 = no higher than 115% of base
    market_position: MarketPosition = MarketPosition.MARKET
    position_pct: float = 0.0   # Override: e.g. 0.03 for +3%, -0.02 for -2%
    min_roi_pct: Optional[float] = None   # For investor: minimum acceptable ROI (e.g. 8.0)

    def clamp(self, price: float, base_price: float) -> float:
        """Clamp price to min/max absolute and relative bounds."""
        p = price
        if self.min_price is not None:
            p = max(p, self.min_price)
        if self.max_price is not None:
            p = min(p, self.max_price)
        if self.min_price_pct_of_base is not None and base_price > 0:
            p = max(p, base_price * self.min_price_pct_of_base)
        if self.max_price_pct_of_base is not None and base_price > 0:
            p = min(p, base_price * self.max_price_pct_of_base)
        return p

    def position_offset(self) -> float:
        """Return the offset to apply for market position (e.g. 0.03 for +3%)."""
        if self.position_pct != 0.0:
            return self.position_pct
        if self.market_position == MarketPosition.AGGRESSIVE:
            return 0.03
        if self.market_position == MarketPosition.CONSERVATIVE:
            return -0.02
        return 0.0
