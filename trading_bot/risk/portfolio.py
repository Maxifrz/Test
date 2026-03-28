from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from trading_bot.config.settings import RiskConfig
from trading_bot.data.models import Position, Signal, TradeStatus


class PortfolioRisk:
    """
    Portfolio-level risk management and circuit breakers.

    Checks:
      1. Max open positions limit
      2. Max portfolio exposure (% of total balance)
      3. Max drawdown circuit breaker (halts trading)
      4. No duplicate symbol + direction positions
    """

    def __init__(self, risk_config: RiskConfig):
        self.cfg = risk_config
        self._positions: Dict[str, Position] = {}  # position_id -> Position
        self._peak_balance: float = 0.0
        self._halted: bool = False

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.status == TradeStatus.OPEN]

    @property
    def is_halted(self) -> bool:
        return self._halted

    def add_position(self, position: Position) -> None:
        self._positions[position.id] = position

    def close_position(self, position_id: str, exit_price: float) -> Optional[Position]:
        pos = self._positions.get(position_id)
        if pos and pos.status == TradeStatus.OPEN:
            pos.close(exit_price)
            logger.info(
                "Position closed: {} {} PnL={:.4f}",
                pos.symbol,
                pos.direction.value,
                pos.pnl,
            )
            return pos
        return None

    def approve(self, signal: Signal, balance: float) -> bool:
        """
        Evaluate whether a new trade is allowed by portfolio risk rules.

        Returns True if the trade is approved.
        """
        if self._halted:
            logger.warning("Trading halted due to max drawdown. Signal rejected.")
            return False

        open_pos = self.open_positions

        # 1. Max open positions
        if len(open_pos) >= self.cfg.max_open_positions:
            logger.debug(
                "Max open positions reached ({}/{}). Signal rejected.",
                len(open_pos),
                self.cfg.max_open_positions,
            )
            return False

        # 2. No duplicate symbol + direction
        for pos in open_pos:
            if pos.symbol == signal.symbol and pos.direction == signal.direction:
                logger.debug(
                    "Duplicate position for {} {}. Signal rejected.",
                    signal.symbol,
                    signal.direction.value,
                )
                return False

        # 3. Max portfolio exposure
        exposed_value = sum(
            pos.entry_price * pos.quantity for pos in open_pos
        )
        if balance > 0 and (exposed_value / balance) >= self.cfg.max_portfolio_exposure:
            logger.debug("Max portfolio exposure reached. Signal rejected.")
            return False

        return True

    def update_drawdown(self, current_balance: float) -> None:
        """
        Track peak balance and trigger circuit breaker if drawdown exceeds limit.
        """
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        if self._peak_balance > 0:
            drawdown = (self._peak_balance - current_balance) / self._peak_balance
            if drawdown >= self.cfg.max_drawdown_pct:
                if not self._halted:
                    logger.error(
                        "MAX DRAWDOWN REACHED ({:.1%}). Trading halted!",
                        drawdown,
                    )
                self._halted = True

    def reset_halt(self) -> None:
        """Manually reset the circuit breaker (use with caution)."""
        self._halted = False
        logger.warning("Trading halt manually reset.")

    def summary(self) -> dict:
        open_pos = self.open_positions
        total_pnl = sum(p.pnl for p in self._positions.values())
        return {
            "open_positions": len(open_pos),
            "total_trades": len(self._positions),
            "total_pnl": round(total_pnl, 4),
            "halted": self._halted,
            "peak_balance": round(self._peak_balance, 4),
        }
