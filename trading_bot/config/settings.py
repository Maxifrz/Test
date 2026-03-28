from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExchangeConfig:
    name: str = "binance"
    testnet: bool = True
    spot_api_key: str = ""
    spot_api_secret: str = ""
    futures_api_key: str = ""
    futures_api_secret: str = ""

    def __post_init__(self):
        self.spot_api_key = os.getenv("BINANCE_SPOT_API_KEY", self.spot_api_key)
        self.spot_api_secret = os.getenv("BINANCE_SPOT_API_SECRET", self.spot_api_secret)
        self.futures_api_key = os.getenv("BINANCE_FUTURES_API_KEY", self.futures_api_key)
        self.futures_api_secret = os.getenv("BINANCE_FUTURES_API_SECRET", self.futures_api_secret)


@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    primary_tf: str = "1h"
    mode: str = "paper"
    candle_limit: int = 500


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.02
    max_portfolio_exposure: float = 0.20
    max_open_positions: int = 3
    max_drawdown_pct: float = 0.10
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    futures_max_leverage: int = 5
    min_confidence: float = 0.55


@dataclass
class MLConfig:
    retrain_interval_hours: int = 24
    min_training_samples: int = 500
    regime_lookback: int = 100
    model_path: str = "models/"


@dataclass
class TrendStrategyConfig:
    ema_fast: int = 21
    ema_slow: int = 50
    adx_threshold: float = 25.0
    adx_period: int = 14


@dataclass
class MeanReversionConfig:
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0


@dataclass
class BreakoutConfig:
    lookback_periods: int = 20
    volume_multiplier: float = 1.5
    atr_period: int = 14


@dataclass
class MomentumConfig:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    roc_period: int = 14


@dataclass
class StrategiesConfig:
    trend: TrendStrategyConfig = field(default_factory=TrendStrategyConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)


@dataclass
class MonitoringConfig:
    refresh_interval: int = 2
    log_level: str = "INFO"
    log_file: str = "logs/trading_bot.log"


@dataclass
class Settings:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    strategies: StrategiesConfig = field(default_factory=StrategiesConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def _merge(dataclass_instance, data: dict):
    """Recursively populate a dataclass from a dict."""
    for key, value in data.items():
        if not hasattr(dataclass_instance, key):
            continue
        current = getattr(dataclass_instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dataclass_instance, key, value)


def load_config(path: str | Path | None = None) -> Settings:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    path = Path(path)
    settings = Settings()
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        for section, values in raw.items():
            if hasattr(settings, section) and isinstance(values, dict):
                _merge(getattr(settings, section), values)
    return settings
