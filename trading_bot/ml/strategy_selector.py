from __future__ import annotations

from typing import Dict

from trading_bot.data.models import MarketRegime


# Default strategy weights per market regime
# Each regime has a weight dict: strategy_name -> weight (0.0 – 1.0, sums to 1.0)
DEFAULT_WEIGHTS: Dict[MarketRegime, Dict[str, float]] = {
    MarketRegime.TRENDING: {
        "trend_following": 0.55,
        "momentum": 0.25,
        "breakout": 0.10,
        "mean_reversion": 0.10,
    },
    MarketRegime.RANGING: {
        "mean_reversion": 0.60,
        "momentum": 0.15,
        "breakout": 0.10,
        "trend_following": 0.15,
    },
    MarketRegime.VOLATILE: {
        "breakout": 0.35,
        "momentum": 0.30,
        "mean_reversion": 0.20,
        "trend_following": 0.15,
    },
    MarketRegime.BREAKOUT: {
        "breakout": 0.50,
        "momentum": 0.25,
        "trend_following": 0.15,
        "mean_reversion": 0.10,
    },
}


class StrategySelector:
    """
    Maps a detected market regime to strategy weights.

    Weights determine how much each strategy's signal contributes to the
    final aggregated signal.  Can be overridden with custom weights.
    """

    def __init__(self, custom_weights: Dict[MarketRegime, Dict[str, float]] | None = None):
        self._weights = custom_weights or DEFAULT_WEIGHTS

    def get_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Return strategy weight dict for the given regime."""
        weights = self._weights.get(regime, DEFAULT_WEIGHTS[MarketRegime.RANGING])
        # Normalise to sum to 1.0
        total = sum(weights.values())
        if total <= 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}

    def top_strategy(self, regime: MarketRegime) -> str:
        """Return the name of the dominant strategy for the regime."""
        weights = self.get_weights(regime)
        return max(weights, key=lambda k: weights[k])

    def update_weights(self, regime: MarketRegime, weights: Dict[str, float]) -> None:
        """Allow runtime updating of weights (e.g. from reinforcement feedback)."""
        self._weights[regime] = weights
