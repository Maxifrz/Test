from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from trading_bot.data.models import MarketRegime, Signal, SignalDirection
from trading_bot.strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """
    EMA crossover strategy with ADX filter and optional multi-TF confirmation.

    Entry conditions (LONG):
      - Fast EMA crosses above slow EMA on current candle
      - ADX > threshold (trend is strong enough)
      - Higher timeframe EMA trend is also bullish (optional confirmation)

    Entry conditions (SHORT):
      - Fast EMA crosses below slow EMA
      - ADX > threshold
      - Higher timeframe bearish

    Stop-loss / Take-profit: ATR-based
    """

    name = "trend_following"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.ema_fast = self.config.get("ema_fast", 21)
        self.ema_slow = self.config.get("ema_slow", 50)
        self.adx_threshold = self.config.get("adx_threshold", 25.0)
        self.sl_mult = self.config.get("sl_mult", 2.0)
        self.tp_mult = self.config.get("tp_mult", 3.0)

    def generate_signal(
        self,
        symbol: str,
        primary_df: pd.DataFrame,
        multi_tf: Optional[Dict[str, pd.DataFrame]] = None,
        regime: MarketRegime = MarketRegime.RANGING,
    ) -> Optional[Signal]:
        if primary_df is None or len(primary_df) < self.ema_slow + 5:
            return None

        fast_col = f"ema_{self.ema_fast}"
        slow_col = f"ema_{self.ema_slow}"

        if fast_col not in primary_df.columns or slow_col not in primary_df.columns:
            return None

        cur = self._latest(primary_df)
        prev = self._prev(primary_df)

        ema_fast_cur = self._col(cur, fast_col)
        ema_slow_cur = self._col(cur, slow_col)
        ema_fast_prev = self._col(prev, fast_col)
        ema_slow_prev = self._col(prev, slow_col)
        adx = self._col(cur, "adx")

        import math
        if any(math.isnan(v) for v in [ema_fast_cur, ema_slow_cur, ema_fast_prev, ema_slow_prev]):
            return None

        # ADX filter
        if not math.isnan(adx) and adx < self.adx_threshold:
            return None

        atr = self._atr(primary_df)
        entry = float(cur["close"])
        direction: Optional[SignalDirection] = None

        # Crossover detection
        bullish_cross = ema_fast_prev <= ema_slow_prev and ema_fast_cur > ema_slow_cur
        bearish_cross = ema_fast_prev >= ema_slow_prev and ema_fast_cur < ema_slow_cur

        if bullish_cross:
            direction = SignalDirection.LONG
        elif bearish_cross:
            direction = SignalDirection.SHORT
        else:
            return None

        # Multi-TF confirmation from higher timeframe
        confidence = 0.65
        if multi_tf:
            for tf, df_higher in multi_tf.items():
                if df_higher.empty:
                    continue
                row = df_higher.iloc[-1]
                hf = self._col(row, fast_col)
                hs = self._col(row, slow_col)
                import math as _math
                if _math.isnan(hf) or _math.isnan(hs):
                    continue
                if direction == SignalDirection.LONG and hf > hs:
                    confidence = min(confidence + 0.10, 0.95)
                elif direction == SignalDirection.SHORT and hf < hs:
                    confidence = min(confidence + 0.10, 0.95)

        sl, tp = self._sl_tp(direction, entry, atr, self.sl_mult, self.tp_mult)

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            strategy_name=self.name,
            regime=regime,
            timeframe=primary_df.index.name or "unknown",
            timestamp=datetime.utcnow(),
            metadata={"adx": adx, "ema_fast": ema_fast_cur, "ema_slow": ema_slow_cur},
        )
