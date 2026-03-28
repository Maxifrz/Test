from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from loguru import logger


def compute_indicators(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Compute all technical indicators on an OHLCV DataFrame.

    Expected columns: open, high, low, close, volume
    Returns the same DataFrame enriched with indicator columns.
    """
    if df.empty or len(df) < 30:
        return df

    cfg = config or {}
    df = df.copy()

    try:
        # ── Trend ──────────────────────────────────────────────────────────────
        ema_fast = cfg.get("ema_fast", 21)
        ema_slow = cfg.get("ema_slow", 50)
        df[f"ema_{ema_fast}"] = ta.ema(df["close"], length=ema_fast)
        df[f"ema_{ema_slow}"] = ta.ema(df["close"], length=ema_slow)
        df["ema_200"] = ta.ema(df["close"], length=200)

        adx_period = cfg.get("adx_period", 14)
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=adx_period)
        if adx_result is not None and not adx_result.empty:
            df["adx"] = adx_result.iloc[:, 0]
            df["di_plus"] = adx_result.iloc[:, 1]
            df["di_minus"] = adx_result.iloc[:, 2]

        # ── Oscillators ────────────────────────────────────────────────────────
        rsi_period = cfg.get("rsi_period", 14)
        df["rsi"] = ta.rsi(df["close"], length=rsi_period)

        macd_result = ta.macd(
            df["close"],
            fast=cfg.get("macd_fast", 12),
            slow=cfg.get("macd_slow", 26),
            signal=cfg.get("macd_signal", 9),
        )
        if macd_result is not None and not macd_result.empty:
            df["macd"] = macd_result.iloc[:, 0]
            df["macd_hist"] = macd_result.iloc[:, 1]
            df["macd_signal"] = macd_result.iloc[:, 2]

        # ── Volatility ─────────────────────────────────────────────────────────
        atr_period = cfg.get("atr_period", 14)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=atr_period)

        bb_period = cfg.get("bb_period", 20)
        bb_std = cfg.get("bb_std", 2.0)
        bb_result = ta.bbands(df["close"], length=bb_period, std=bb_std)
        if bb_result is not None and not bb_result.empty:
            df["bb_lower"] = bb_result.iloc[:, 0]
            df["bb_mid"] = bb_result.iloc[:, 1]
            df["bb_upper"] = bb_result.iloc[:, 2]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ── Momentum ───────────────────────────────────────────────────────────
        roc_period = cfg.get("roc_period", 14)
        df["roc"] = ta.roc(df["close"], length=roc_period)

        # ── Volume ─────────────────────────────────────────────────────────────
        df["volume_sma"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, float("nan"))

        # ── Stochastic ─────────────────────────────────────────────────────────
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

    except Exception as exc:
        logger.warning("Indicator computation error: {}", exc)

    return df
