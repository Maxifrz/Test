from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from trading_bot.data.models import MarketRegime, Signal, SignalDirection
from trading_bot.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    MACD + Rate of Change momentum strategy.

    Entry conditions (LONG):
      - MACD histogram crosses from negative to positive (momentum turning bullish)
      - ROC > 0 (positive momentum over N periods)
      - Optional: RSI between 40–65 (not overbought/oversold extremes)

    Entry conditions (SHORT):
      - MACD histogram crosses from positive to negative
      - ROC < 0

    Best suited for TRENDING and early BREAKOUT regimes.
    """

    name = "momentum"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.sl_mult = self.config.get("sl_mult", 2.0)
        self.tp_mult = self.config.get("tp_mult", 2.5)
        self.rsi_min = self.config.get("rsi_min", 35.0)
        self.rsi_max = self.config.get("rsi_max", 65.0)

    def generate_signal(
        self,
        symbol: str,
        primary_df: pd.DataFrame,
        multi_tf: Optional[Dict[str, pd.DataFrame]] = None,
        regime: MarketRegime = MarketRegime.RANGING,
    ) -> Optional[Signal]:
        if primary_df is None or len(primary_df) < 35:
            return None

        required = ["macd_hist", "roc"]
        cur = self._latest(primary_df)
        prev = self._prev(primary_df)

        macd_hist_cur = self._col(cur, "macd_hist")
        macd_hist_prev = self._col(prev, "macd_hist")
        roc = self._col(cur, "roc")
        rsi = self._col(cur, "rsi")
        close = float(cur["close"])
        atr = self._atr(primary_df)

        if math.isnan(macd_hist_cur) or math.isnan(macd_hist_prev) or math.isnan(roc):
            return None

        direction: Optional[SignalDirection] = None
        confidence = 0.58

        # MACD histogram crossover (zero-line cross)
        bullish_cross = macd_hist_prev <= 0 < macd_hist_cur
        bearish_cross = macd_hist_prev >= 0 > macd_hist_cur

        if bullish_cross and roc > 0:
            direction = SignalDirection.LONG
        elif bearish_cross and roc < 0:
            direction = SignalDirection.SHORT
        else:
            return None

        # RSI sanity check – avoid entering at extremes
        if not math.isnan(rsi):
            if direction == SignalDirection.LONG and rsi > self.rsi_max:
                return None
            if direction == SignalDirection.SHORT and rsi < self.rsi_min:
                return None
            # Additional confidence if RSI is in "sweet spot"
            if self.rsi_min < rsi < self.rsi_max:
                confidence += 0.07

        # Bonus in trending regime
        if regime == MarketRegime.TRENDING:
            confidence = min(confidence + 0.10, 0.90)

        # Multi-TF momentum confirmation
        if multi_tf:
            for tf, df_higher in multi_tf.items():
                if df_higher.empty or len(df_higher) < 2:
                    continue
                h_cur = df_higher.iloc[-1]
                h_prev = df_higher.iloc[-2]
                h_hist_cur = self._col(h_cur, "macd_hist")
                h_hist_prev = self._col(h_prev, "macd_hist")
                if not (math.isnan(h_hist_cur) or math.isnan(h_hist_prev)):
                    if direction == SignalDirection.LONG and h_hist_cur > h_hist_prev:
                        confidence = min(confidence + 0.05, 0.90)
                    elif direction == SignalDirection.SHORT and h_hist_cur < h_hist_prev:
                        confidence = min(confidence + 0.05, 0.90)

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
            metadata={"macd_hist": macd_hist_cur, "roc": roc, "rsi": rsi},
        )
