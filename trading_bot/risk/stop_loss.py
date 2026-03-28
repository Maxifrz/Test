from __future__ import annotations

from typing import Tuple

import pandas as pd

from trading_bot.data.models import Signal, SignalDirection
from trading_bot.config.settings import RiskConfig


class ATRStopLoss:
    """
    Calculates ATR-based stop-loss and take-profit levels.

    SL = entry ± (ATR * sl_multiplier)
    TP = entry ± (ATR * tp_multiplier)

    Can also refine SL/TP on an existing signal.
    """

    def __init__(self, risk_config: RiskConfig):
        self.cfg = risk_config

    def calculate(
        self,
        direction: SignalDirection,
        entry: float,
        atr: float,
        sl_multiplier: float | None = None,
        tp_multiplier: float | None = None,
    ) -> Tuple[float, float]:
        """
        Returns (stop_loss, take_profit).
        """
        sl_mult = sl_multiplier if sl_multiplier is not None else self.cfg.atr_sl_multiplier
        tp_mult = tp_multiplier if tp_multiplier is not None else self.cfg.atr_tp_multiplier

        if direction == SignalDirection.LONG:
            sl = entry - atr * sl_mult
            tp = entry + atr * tp_mult
        else:
            sl = entry + atr * sl_mult
            tp = entry - atr * tp_mult

        return sl, tp

    def refine_signal(self, signal: Signal, df: pd.DataFrame) -> Signal:
        """
        Recompute SL/TP for an existing signal using the latest ATR.
        """
        if df.empty or "atr" not in df.columns:
            return signal

        atr = float(df["atr"].iloc[-1])
        if pd.isna(atr) or atr <= 0:
            return signal

        sl, tp = self.calculate(signal.direction, signal.entry_price, atr)
        signal.stop_loss = sl
        signal.take_profit = tp
        return signal

    def trailing_stop(
        self,
        direction: SignalDirection,
        current_price: float,
        current_sl: float,
        atr: float,
        trail_multiplier: float = 1.5,
    ) -> float:
        """
        Update a trailing stop-loss.
        For LONG: only move SL up when price moves up.
        For SHORT: only move SL down when price moves down.
        """
        new_sl = current_price - atr * trail_multiplier if direction == SignalDirection.LONG \
            else current_price + atr * trail_multiplier

        if direction == SignalDirection.LONG:
            return max(current_sl, new_sl)
        return min(current_sl, new_sl)
