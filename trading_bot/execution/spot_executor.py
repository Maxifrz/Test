from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from loguru import logger

from trading_bot.data.models import (
    MarketType,
    Position,
    Signal,
    SignalDirection,
    Trade,
    TradeStatus,
)
from trading_bot.execution.broker import Broker


class SpotExecutor:
    """
    Handles spot market order execution.

    Order flow:
      1. Attempt limit order at signal entry price
      2. If not filled within timeout, cancel and fall back to market order
      3. Track open positions and monitor SL/TP
    """

    LIMIT_TIMEOUT_SECONDS = 30
    SLIPPAGE_TOLERANCE = 0.003  # 0.3%

    def __init__(self, broker: Broker, mode: str = "paper"):
        self.broker = broker
        self.mode = mode  # "paper" | "live"

    async def execute(self, signal: Signal, quantity: float) -> Optional[Position]:
        """
        Execute a trade signal. Returns an open Position or None on failure.
        """
        if not signal.is_valid:
            logger.warning("Invalid signal, skipping execution: {}", signal)
            return None

        if self.mode == "paper":
            return self._paper_execute(signal, quantity)

        side = "buy" if signal.direction == SignalDirection.LONG else "sell"
        entry = signal.entry_price

        # Try limit order first
        order = self.broker.place_limit_order(signal.symbol, side, quantity, entry)
        if order is None:
            return None

        # Wait for fill
        filled_order = await self._wait_for_fill(order["id"], signal.symbol)
        if filled_order is None or filled_order.get("status") != "closed":
            logger.warning("Limit order not filled, falling back to market order")
            self.broker.cancel_order(order["id"], signal.symbol)
            order = self.broker.place_market_order(signal.symbol, side, quantity)
            if order is None:
                return None
            entry = float(order.get("average", entry))
        else:
            entry = float(filled_order.get("average", entry))

        # Check slippage
        slippage = abs(entry - signal.entry_price) / signal.entry_price
        if slippage > self.SLIPPAGE_TOLERANCE:
            logger.warning(
                "High slippage detected: {:.2%} for {}. Proceeding anyway.",
                slippage,
                signal.symbol,
            )

        return self._build_position(signal, quantity, entry)

    def _paper_execute(self, signal: Signal, quantity: float) -> Position:
        """Simulate order execution without placing a real order."""
        logger.info(
            "[PAPER] {} {} {} qty={:.6f} entry={:.4f} sl={:.4f} tp={:.4f}",
            signal.direction.value,
            signal.symbol,
            signal.strategy_name,
            quantity,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
        )
        return self._build_position(signal, quantity, signal.entry_price)

    def _build_position(self, signal: Signal, quantity: float, entry: float) -> Position:
        return Position(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            direction=signal.direction,
            market_type=MarketType.SPOT,
            entry_price=entry,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            leverage=1,
            opened_at=datetime.utcnow(),
            strategy_name=signal.strategy_name,
        )

    async def _wait_for_fill(self, order_id: str, symbol: str) -> Optional[dict]:
        """Poll order status until filled or timeout."""
        deadline = asyncio.get_event_loop().time() + self.LIMIT_TIMEOUT_SECONDS
        while asyncio.get_event_loop().time() < deadline:
            order = self.broker.fetch_order(order_id, symbol)
            if order and order.get("status") == "closed":
                return order
            await asyncio.sleep(2)
        return None

    def check_exit(self, position: Position, current_price: float) -> bool:
        """
        Check if position should be closed (SL or TP hit).
        Returns True if position should be closed.
        """
        if position.direction == SignalDirection.LONG:
            if current_price <= position.stop_loss:
                logger.info("SL hit for {} @ {:.4f}", position.symbol, current_price)
                return True
            if current_price >= position.take_profit:
                logger.info("TP hit for {} @ {:.4f}", position.symbol, current_price)
                return True
        else:
            if current_price >= position.stop_loss:
                logger.info("SL hit for {} @ {:.4f}", position.symbol, current_price)
                return True
            if current_price <= position.take_profit:
                logger.info("TP hit for {} @ {:.4f}", position.symbol, current_price)
                return True
        return False

    def build_trade_record(self, position: Position, regime: str = "") -> Trade:
        return Trade(
            position_id=position.id,
            symbol=position.symbol,
            direction=position.direction.value,
            market_type=position.market_type.value,
            entry_price=position.entry_price,
            exit_price=position.exit_price or position.entry_price,
            quantity=position.quantity,
            leverage=position.leverage,
            pnl=position.pnl,
            strategy_name=position.strategy_name,
            opened_at=position.opened_at,
            closed_at=position.closed_at or datetime.utcnow(),
            regime=regime,
        )
