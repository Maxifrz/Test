from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from trading_bot.data.models import MarketRegime, Signal, SignalDirection
from trading_bot.strategies.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """
    Support/Resistance breakout strategy with volume confirmation.

    Entry conditions (LONG):
      - Close breaks above the rolling N-period high (resistance)
      - Volume is significantly above average (confirms breakout)

    Entry conditions (SHORT):
      - Close breaks below the rolling N-period low (support)
      - Volume confirmation

    Best suited for BREAKOUT and VOLATILE regimes.
    """

    name = "breakout"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.lookback = self.config.get("lookback_periods", 20)
        self.volume_mult = self.config.get("volume_multiplier", 1.5)
        self.sl_mult = self.config.get("sl_mult", 1.5)
        self.tp_mult = self.config.get("tp_mult", 2.5)
        self.retest_tolerance = self.config.get("retest_tolerance", 0.002)  # 0.2%

    def generate_signal(
        self,
        symbol: str,
        primary_df: pd.DataFrame,
        multi_tf: Optional[Dict[str, pd.DataFrame]] = None,
        regime: MarketRegime = MarketRegime.RANGING,
    ) -> Optional[Signal]:
        if primary_df is None or len(primary_df) < self.lookback + 5:
            return None

        cur = self._latest(primary_df)
        close = float(cur["close"])
        volume = float(cur["volume"]) if "volume" in cur.index else 0.0
        atr = self._atr(primary_df)

        # Key levels: rolling high/low over lookback (excluding current candle)
        lookback_slice = primary_df.iloc[-(self.lookback + 1):-1]
        if lookback_slice.empty:
            return None

        resistance = float(lookback_slice["high"].max())
        support = float(lookback_slice["low"].min())

        # Volume confirmation
        vol_sma = self._col(cur, "volume_sma")
        vol_ratio = self._col(cur, "volume_ratio")
        if not math.isnan(vol_ratio):
            volume_confirmed = vol_ratio >= self.volume_mult
        elif not math.isnan(vol_sma) and vol_sma > 0:
            volume_confirmed = volume >= vol_sma * self.volume_mult
        else:
            volume_confirmed = True  # no volume data, skip filter

        direction: Optional[SignalDirection] = None
        confidence = 0.60

        # Bullish breakout
        if close > resistance * (1 + self.retest_tolerance):
            direction = SignalDirection.LONG
            if volume_confirmed:
                confidence += 0.15

        # Bearish breakout
        elif close < support * (1 - self.retest_tolerance):
            direction = SignalDirection.SHORT
            if volume_confirmed:
                confidence += 0.15

        if direction is None:
            return None

        # Bonus in breakout/volatile regime
        if regime in (MarketRegime.BREAKOUT, MarketRegime.VOLATILE):
            confidence = min(confidence + 0.10, 0.92)

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
            metadata={
                "resistance": resistance,
                "support": support,
                "volume_ratio": float(vol_ratio) if not math.isnan(vol_ratio) else None,
            },
        )
