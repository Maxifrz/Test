from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from trading_bot.data.models import MarketRegime, Signal, SignalDirection
from trading_bot.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    RSI extreme + Bollinger Band strategy.

    Entry conditions (LONG):
      - RSI < oversold threshold (e.g. 30)
      - Price near or below lower Bollinger Band
      - Bollinger Band is NOT in squeeze (some width exists)

    Entry conditions (SHORT):
      - RSI > overbought threshold (e.g. 70)
      - Price near or above upper Bollinger Band

    Best suited for RANGING regimes.
    """

    name = "mean_reversion"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.rsi_oversold = self.config.get("rsi_oversold", 30.0)
        self.rsi_overbought = self.config.get("rsi_overbought", 70.0)
        self.sl_mult = self.config.get("sl_mult", 1.5)
        self.tp_mult = self.config.get("tp_mult", 2.0)
        self.bb_touch_tolerance = self.config.get("bb_touch_tolerance", 0.005)  # 0.5%

    def generate_signal(
        self,
        symbol: str,
        primary_df: pd.DataFrame,
        multi_tf: Optional[Dict[str, pd.DataFrame]] = None,
        regime: MarketRegime = MarketRegime.RANGING,
    ) -> Optional[Signal]:
        if primary_df is None or len(primary_df) < 30:
            return None

        required = ["rsi", "bb_lower", "bb_upper", "bb_mid", "bb_width", "atr"]
        cur = self._latest(primary_df)
        if any(c not in cur.index or math.isnan(float(cur[c])) for c in required if c in primary_df.columns):
            pass  # proceed with available cols

        rsi = self._col(cur, "rsi")
        close = float(cur["close"])
        bb_lower = self._col(cur, "bb_lower")
        bb_upper = self._col(cur, "bb_upper")
        bb_width = self._col(cur, "bb_width")
        atr = self._atr(primary_df)

        if math.isnan(rsi) or math.isnan(bb_lower) or math.isnan(bb_upper):
            return None

        # Avoid trading in extreme squeeze (no range to revert to)
        if not math.isnan(bb_width) and bb_width < 0.01:
            return None

        direction: Optional[SignalDirection] = None
        confidence = 0.60

        # LONG: oversold + price touching lower BB
        if rsi < self.rsi_oversold and close <= bb_lower * (1 + self.bb_touch_tolerance):
            direction = SignalDirection.LONG
            # Extra confidence if RSI is deeply oversold
            if rsi < self.rsi_oversold - 10:
                confidence += 0.10

        # SHORT: overbought + price touching upper BB
        elif rsi > self.rsi_overbought and close >= bb_upper * (1 - self.bb_touch_tolerance):
            direction = SignalDirection.SHORT
            if rsi > self.rsi_overbought + 10:
                confidence += 0.10

        if direction is None:
            return None

        # Bonus confidence in a ranging regime
        if regime == MarketRegime.RANGING:
            confidence = min(confidence + 0.08, 0.92)

        sl, tp = self._sl_tp(direction, close, atr, self.sl_mult, self.tp_mult)

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            strategy_name=self.name,
            regime=regime,
            timeframe=primary_df.index.name or "unknown",
            timestamp=datetime.utcnow(),
            metadata={"rsi": rsi, "bb_lower": bb_lower, "bb_upper": bb_upper},
        )
