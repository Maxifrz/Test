from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

from trading_bot.data.models import MarketRegime, Signal, SignalDirection


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = "base"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        primary_df: pd.DataFrame,
        multi_tf: Optional[Dict[str, pd.DataFrame]] = None,
        regime: MarketRegime = MarketRegime.RANGING,
    ) -> Optional[Signal]:
        """
        Analyse the DataFrame(s) and return a Signal or None.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT"
            primary_df: Indicator-enriched OHLCV DataFrame for the primary timeframe
            multi_tf: Optional dict of {timeframe: DataFrame} for confirmation
            regime: Current market regime (may influence strategy behaviour)

        Returns:
            Signal if an entry opportunity is found, else None.
        """

    def _latest(self, df: pd.DataFrame) -> pd.Series:
        return df.iloc[-1]

    def _prev(self, df: pd.DataFrame, n: int = 1) -> pd.Series:
        return df.iloc[-(1 + n)]

    def _col(self, row: pd.Series, name: str, default: float = float("nan")) -> float:
        return float(row[name]) if name in row.index and pd.notna(row[name]) else default

    def _atr(self, df: pd.DataFrame) -> float:
        if "atr" in df.columns:
            val = df["atr"].iloc[-1]
            if pd.notna(val):
                return float(val)
        # fallback: simple range
        return float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])

    def _sl_tp(
        self,
        direction: SignalDirection,
        entry: float,
        atr: float,
        sl_mult: float = 2.0,
        tp_mult: float = 3.0,
    ) -> tuple[float, float]:
        if direction == SignalDirection.LONG:
            sl = entry - atr * sl_mult
            tp = entry + atr * tp_mult
        else:
            sl = entry + atr * sl_mult
            tp = entry - atr * tp_mult
        return sl, tp
