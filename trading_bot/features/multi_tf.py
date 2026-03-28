from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from trading_bot.features.indicators import compute_indicators


class MultiTFFeatures:
    """
    Maintains indicator-enriched DataFrames for multiple timeframes per symbol.
    Provides a unified feature dict for any symbol at the current moment.
    """

    # Maps timeframe string to minutes for ordering
    TF_MINUTES: Dict[str, int] = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "8h": 480,
        "1d": 1440, "1w": 10080,
    }

    def __init__(self, timeframes: List[str], indicator_config: Optional[dict] = None):
        self.timeframes = sorted(timeframes, key=lambda t: self.TF_MINUTES.get(t, 0))
        self.indicator_config = indicator_config or {}
        # symbol -> timeframe -> DataFrame
        self._frames: Dict[str, Dict[str, pd.DataFrame]] = {}

    def update(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Update the cached DataFrame for a symbol/timeframe."""
        if df.empty:
            return
        enriched = compute_indicators(df, self.indicator_config)
        if symbol not in self._frames:
            self._frames[symbol] = {}
        self._frames[symbol][timeframe] = enriched

    def get(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Return indicator-enriched DataFrames keyed by timeframe."""
        return self._frames.get(symbol, {})

    def get_latest_row(self, symbol: str, timeframe: str) -> Optional[pd.Series]:
        """Return the most recent row for a symbol/timeframe."""
        frames = self._frames.get(symbol, {})
        df = frames.get(timeframe)
        if df is None or df.empty:
            return None
        return df.iloc[-1]

    def build_feature_vector(self, symbol: str, primary_tf: str) -> Dict[str, float]:
        """
        Build a flat feature dict combining key indicators across all timeframes.
        Used as input to the ML regime classifier.
        """
        features: Dict[str, float] = {}
        for tf in self.timeframes:
            row = self.get_latest_row(symbol, tf)
            if row is None:
                continue
            prefix = tf.replace("m", "m").replace("h", "h").replace("d", "d")
            for col in ["rsi", "adx", "atr", "roc", "macd_hist", "bb_width", "volume_ratio"]:
                if col in row.index and pd.notna(row[col]):
                    features[f"{prefix}_{col}"] = float(row[col])

        # Trend alignment across timeframes
        close_prices = {}
        for tf in self.timeframes:
            row = self.get_latest_row(symbol, tf)
            if row is not None and "close" in row.index:
                close_prices[tf] = float(row["close"])

        if len(close_prices) >= 2:
            tfs_sorted = sorted(close_prices.keys(), key=lambda t: self.TF_MINUTES.get(t, 0))
            short_tf = tfs_sorted[0]
            long_tf = tfs_sorted[-1]
            if close_prices[long_tf] != 0:
                features["price_change_ratio"] = (
                    close_prices[short_tf] / close_prices[long_tf] - 1.0
                )

        return features

    def is_ready(self, symbol: str, required_tfs: Optional[List[str]] = None) -> bool:
        """Check if we have data for all required timeframes."""
        tfs = required_tfs or self.timeframes
        frames = self._frames.get(symbol, {})
        return all(tf in frames and not frames[tf].empty for tf in tfs)
