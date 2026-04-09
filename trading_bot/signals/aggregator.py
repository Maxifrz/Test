from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from trading_bot.data.models import MarketRegime, Signal, SignalDirection


class SignalAggregator:
    """
    Combines signals from multiple strategies into a single final signal.

    Aggregation logic:
      1. Each strategy produces an optional Signal
      2. Strategy weights (from StrategySelector) are applied to confidence scores
      3. Weighted vote determines direction (LONG / SHORT / FLAT)
      4. Final confidence = weighted average of individual confidences
      5. SL/TP are taken from the highest-confidence individual signal
    """

    def aggregate(
        self,
        signals: Dict[str, Optional[Signal]],
        weights: Dict[str, float],
        min_confidence: float = 0.55,
    ) -> Optional[Signal]:
        """
        Args:
            signals: {strategy_name: Signal or None}
            weights: {strategy_name: weight}
            min_confidence: discard final signal below this threshold

        Returns:
            Aggregated Signal or None
        """
        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        candidates: List[Signal] = []

        for strategy_name, signal in signals.items():
            if signal is None:
                continue
            w = weights.get(strategy_name, 0.0)
            if w <= 0:
                continue

            if signal.direction == SignalDirection.LONG:
                long_score += signal.confidence * w
            elif signal.direction == SignalDirection.SHORT:
                short_score += signal.confidence * w

            total_weight += w
            candidates.append(signal)

        if not candidates or total_weight == 0:
            return None

        # Normalise scores
        long_score /= total_weight
        short_score /= total_weight

        if long_score > short_score and long_score >= min_confidence:
            direction = SignalDirection.LONG
            confidence = long_score
        elif short_score > long_score and short_score >= min_confidence:
            direction = SignalDirection.SHORT
            confidence = short_score
        else:
            return None  # No clear consensus

        # Pick the best individual signal for SL/TP/metadata
        best_signal = max(
            [s for s in candidates if s.direction == direction],
            key=lambda s: s.confidence,
        )

        regime = best_signal.regime
        strategy_names = [s.strategy_name for s in candidates if s.direction == direction]
        logger.debug(
            "Aggregated signal: {} {} conf={:.2f} strategies={}",
            direction.value,
            best_signal.symbol,
            confidence,
            strategy_names,
        )

        return Signal(
            symbol=best_signal.symbol,
            direction=direction,
            confidence=round(confidence, 4),
            entry_price=best_signal.entry_price,
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
            strategy_name=",".join(strategy_names),
            regime=regime,
            timeframe=best_signal.timeframe,
            timestamp=datetime.utcnow(),
            metadata={
                "long_score": round(long_score, 4),
                "short_score": round(short_score, 4),
                "contributing_strategies": strategy_names,
            },
        )
