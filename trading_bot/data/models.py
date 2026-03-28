from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"


class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class MarketType(Enum):
    SPOT = "spot"
    FUTURES = "futures"


@dataclass
class OHLCV:
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_ccxt(cls, row: list, symbol: str, timeframe: str) -> "OHLCV":
        return cls(
            timestamp=datetime.utcfromtimestamp(row[0] / 1000),
            symbol=symbol,
            timeframe=timeframe,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )


@dataclass
class Signal:
    symbol: str
    direction: SignalDirection
    confidence: float  # 0.0 – 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    regime: MarketRegime
    timeframe: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        if self.direction == SignalDirection.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        return reward / risk if risk > 0 else 0.0

    @property
    def is_valid(self) -> bool:
        return (
            self.direction != SignalDirection.FLAT
            and self.confidence > 0
            and self.stop_loss > 0
            and self.take_profit > 0
            and self.risk_reward >= 1.0
        )


@dataclass
class Position:
    id: str
    symbol: str
    direction: SignalDirection
    market_type: MarketType
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    leverage: int = 1
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    strategy_name: str = ""

    @property
    def unrealized_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.direction == SignalDirection.LONG:
            return (self.exit_price - self.entry_price) * self.quantity * self.leverage
        return (self.entry_price - self.exit_price) * self.quantity * self.leverage

    def close(self, exit_price: float) -> None:
        self.exit_price = exit_price
        self.closed_at = datetime.utcnow()
        self.status = TradeStatus.CLOSED
        if self.direction == SignalDirection.LONG:
            self.pnl = (exit_price - self.entry_price) * self.quantity * self.leverage
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity * self.leverage


@dataclass
class Trade:
    position_id: str
    symbol: str
    direction: str
    market_type: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    pnl: float
    strategy_name: str
    opened_at: datetime
    closed_at: datetime
    regime: str = ""
