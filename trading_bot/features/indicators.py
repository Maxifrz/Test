from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ── Pure numpy/pandas indicator implementations ──────────────────────────────


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, histogram, signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _bbands(series: pd.Series, period: int = 20, std: float = 2.0):
    mid = _sma(series, period)
    sigma = series.rolling(window=period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    return lower, mid, upper


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus = np.where((high - prev_high) > (prev_low - low), (high - prev_high).clip(lower=0), 0.0)
    dm_minus = np.where((prev_low - low) > (high - prev_high), (prev_low - low).clip(lower=0), 0.0)

    dm_plus_s = pd.Series(dm_plus, index=high.index).ewm(com=period - 1, adjust=False).mean()
    dm_minus_s = pd.Series(dm_minus, index=high.index).ewm(com=period - 1, adjust=False).mean()
    atr_s = tr.ewm(com=period - 1, adjust=False).mean()

    di_plus = 100 * dm_plus_s / atr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus_s / atr_s.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, adjust=False).mean()
    return adx, di_plus, di_minus


def _roc(series: pd.Series, period: int = 14) -> pd.Series:
    return series.pct_change(periods=period) * 100


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = _sma(stoch_k, d)
    return stoch_k, stoch_d


# ── Main entry point ─────────────────────────────────────────────────────────


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
        df[f"ema_{ema_fast}"] = _ema(df["close"], ema_fast)
        df[f"ema_{ema_slow}"] = _ema(df["close"], ema_slow)
        df["ema_200"] = _ema(df["close"], 200)

        adx_period = cfg.get("adx_period", 14)
        adx, di_plus, di_minus = _adx(df["high"], df["low"], df["close"], adx_period)
        df["adx"] = adx
        df["di_plus"] = di_plus
        df["di_minus"] = di_minus

        # ── Oscillators ────────────────────────────────────────────────────────
        rsi_period = cfg.get("rsi_period", 14)
        df["rsi"] = _rsi(df["close"], rsi_period)

        macd_line, macd_hist, macd_sig = _macd(
            df["close"],
            fast=cfg.get("macd_fast", 12),
            slow=cfg.get("macd_slow", 26),
            signal=cfg.get("macd_signal", 9),
        )
        df["macd"] = macd_line
        df["macd_hist"] = macd_hist
        df["macd_signal"] = macd_sig

        # ── Volatility ─────────────────────────────────────────────────────────
        atr_period = cfg.get("atr_period", 14)
        df["atr"] = _atr(df["high"], df["low"], df["close"], atr_period)

        bb_period = cfg.get("bb_period", 20)
        bb_std = cfg.get("bb_std", 2.0)
        bb_lower, bb_mid, bb_upper = _bbands(df["close"], bb_period, bb_std)
        df["bb_lower"] = bb_lower
        df["bb_mid"] = bb_mid
        df["bb_upper"] = bb_upper
        df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

        # ── Momentum ───────────────────────────────────────────────────────────
        roc_period = cfg.get("roc_period", 14)
        df["roc"] = _roc(df["close"], roc_period)

        # ── Volume ─────────────────────────────────────────────────────────────
        df["volume_sma"] = _sma(df["volume"], 20)
        df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)

        # ── Stochastic ─────────────────────────────────────────────────────────
        stoch_k, stoch_d = _stoch(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d

    except Exception as exc:
        logger.warning("Indicator computation error: {}", exc)

    return df
