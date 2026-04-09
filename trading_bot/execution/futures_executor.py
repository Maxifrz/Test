from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from loguru import logger

from trading_bot.config.settings import RiskConfig
from trading_bot.data.models import (
    MarketType,
    Position,
    Signal,
    SignalDirection,
    Trade,
)
from trading_bot.execution.broker import Broker


class FuturesExecutor:
    """
    Handles futures market order execution with leverage management.

    Additional steps vs spot:
      - Set leverage before opening position
      - Set margin mode (isolated recommended for risk control)
      - Handle both long and short positions natively
    """

    LIMIT_TIMEOUT_SECONDS = 30
    SLIPPAGE_TOLERANCE = 0.003

    def __init__(self, broker: Broker, risk_config: RiskConfig, mode: str = "paper"):
        self.broker = broker
        self.risk_cfg = risk_config
        self.mode = mode

    async def execute(
        self,
        signal: Signal,
        quantity: float,
        leverage: Optional[int] = None,
    ) -> Optional[Position]:
        """Execute a futures trade signal."""
        if not signal.is_valid:
            logger.warning("Invalid signal for futures execution")
            return None

        lev = min(leverage or self.risk_cfg.futures_max_leverage, self.risk_cfg.futures_max_leverage)

        if self.mode == "paper":
            return self._paper_execute(signal, quantity, lev)

        # Set leverage and margin mode
        self._configure_futures(signal.symbol, lev)

        side = "buy" if signal.direction == SignalDirection.LONG else "sell"
        entry = signal.entry_price

        order = self.broker.place_limit_order(signal.symbol, side, quantity, entry)
        if order is None:
            return None

        filled = await self._wait_for_fill(order["id"], signal.symbol)
        if filled is None or filled.get("status") != "closed":
            self.broker.cancel_order(order["id"], signal.symbol)
            order = self.broker.place_market_order(signal.symbol, side, quantity)
            if order is None:
                return None
            entry = float(order.get("average", entry))
        else:
            entry = float(filled.get("average", entry))

        return self._build_position(signal, quantity, entry, lev)

    def _configure_futures(self, symbol: str, leverage: int) -> None:
        """Set leverage and isolated margin mode on the exchange."""
        try:
            exchange = self.broker._exchange
            if hasattr(exchange, "set_leverage"):
                exchange.set_leverage(leverage, symbol)
            if hasattr(exchange, "set_margin_mode"):
                exchange.set_margin_mode("isolated", symbol)
            logger.debug("Futures configured: {} leverage={}x isolated", symbol, leverage)
        except Exception as exc:
            logger.warning("Could not configure futures settings: {}", exc)

    def _paper_execute(self, signal: Signal, quantity: float, leverage: int) -> Position:
        logger.info(
            "[PAPER FUTURES] {} {} {}x qty={:.6f} entry={:.4f} sl={:.4f} tp={:.4f}",
            signal.direction.value,
            signal.symbol,
            leverage,
            quantity,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
        )
        return self._build_position(signal, quantity, signal.entry_price, leverage)

    def _build_position(
        self, signal: Signal, quantity: float, entry: float, leverage: int
    ) -> Position:
        return Position(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            direction=signal.direction,
            market_type=MarketType.FUTURES,
            entry_price=entry,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            leverage=leverage,
            opened_at=datetime.utcnow(),
            strategy_name=signal.strategy_name,
        )

    async def _wait_for_fill(self, order_id: str, symbol: str) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + self.LIMIT_TIMEOUT_SECONDS
        while asyncio.get_event_loop().time() < deadline:
            order = self.broker.fetch_order(order_id, symbol)
            if order and order.get("status") == "closed":
                return order
            await asyncio.sleep(2)
        return None

    def check_exit(self, position: Position, current_price: float) -> bool:
        if position.direction == SignalDirection.LONG:
            return current_price <= position.stop_loss or current_price >= position.take_profit
        return current_price >= position.stop_loss or current_price <= position.take_profit

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
