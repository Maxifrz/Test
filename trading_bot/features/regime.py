from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from trading_bot.data.models import MarketRegime


def build_regime_features(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Build a feature vector for regime classification from an enriched OHLCV DataFrame.
    Assumes compute_indicators() has already been applied.
    """
    if df.empty or len(df) < 20:
        return {}

    window = df.iloc[-lookback:] if len(df) >= lookback else df
    latest = df.iloc[-1]
    features: Dict[str, float] = {}

    # ── Trend strength ──────────────────────────────────────────────────────
    if "adx" in df.columns and pd.notna(latest.get("adx")):
        features["adx"] = float(latest["adx"])

    # ── Volatility (normalised ATR) ─────────────────────────────────────────
    if "atr" in df.columns and pd.notna(latest.get("atr")) and latest["close"] > 0:
        features["atr_pct"] = float(latest["atr"]) / float(latest["close"])

    # ── Bollinger Band width (squeeze indicator) ────────────────────────────
    if "bb_width" in df.columns and pd.notna(latest.get("bb_width")):
        features["bb_width"] = float(latest["bb_width"])

    # ── RSI ─────────────────────────────────────────────────────────────────
    if "rsi" in df.columns and pd.notna(latest.get("rsi")):
        features["rsi"] = float(latest["rsi"])
        features["rsi_extreme"] = float(abs(latest["rsi"] - 50) / 50)

    # ── Rate of Change ──────────────────────────────────────────────────────
    if "roc" in df.columns and pd.notna(latest.get("roc")):
        features["roc"] = float(latest["roc"])
        features["roc_abs"] = float(abs(latest["roc"]))

    # ── Volume spike ────────────────────────────────────────────────────────
    if "volume_ratio" in df.columns and pd.notna(latest.get("volume_ratio")):
        features["volume_ratio"] = float(latest["volume_ratio"])

    # ── Price returns volatility (realised vol over window) ─────────────────
    if "close" in window.columns and len(window) > 5:
        returns = window["close"].pct_change().dropna()
        if not returns.empty:
            features["realised_vol"] = float(returns.std())
            features["skewness"] = float(returns.skew())

    # ── EMA trend direction ─────────────────────────────────────────────────
    if "ema_21" in df.columns and "ema_50" in df.columns:
        ema_fast = latest.get("ema_21")
        ema_slow = latest.get("ema_50")
        if pd.notna(ema_fast) and pd.notna(ema_slow) and float(ema_slow) > 0:
            features["ema_ratio"] = float(ema_fast) / float(ema_slow) - 1.0

    # ── MACD histogram sign and magnitude ──────────────────────────────────
    if "macd_hist" in df.columns and pd.notna(latest.get("macd_hist")):
        features["macd_hist"] = float(latest["macd_hist"])
        features["macd_hist_abs"] = float(abs(latest["macd_hist"]))

    return features


def heuristic_regime(features: Dict[str, float]) -> MarketRegime:
    """
    Simple rule-based regime detection used as fallback when the ML model
    is not yet trained.
    """
    adx = features.get("adx", 20)
    atr_pct = features.get("atr_pct", 0.01)
    bb_width = features.get("bb_width", 0.05)
    volume_ratio = features.get("volume_ratio", 1.0)
    roc_abs = features.get("roc_abs", 0.0)

    # Breakout: volume spike + price move + volatility expanding
    if volume_ratio > 1.8 and bb_width > 0.08 and roc_abs > 2.0:
        return MarketRegime.BREAKOUT

    # Trending: strong ADX + clear directional move
    if adx > 30 and roc_abs > 1.5:
        return MarketRegime.TRENDING

    # Volatile: high ATR but no clear trend
    if atr_pct > 0.025 and adx < 25:
        return MarketRegime.VOLATILE

    # Ranging: low ADX, tight bands
    return MarketRegime.RANGING
