from __future__ import annotations

import math
from typing import Optional

from loguru import logger

from trading_bot.data.models import Signal, SignalDirection
from trading_bot.config.settings import RiskConfig


class PositionSizer:
    """
    Calculates trade position size using Fixed Fractional or Kelly Criterion.

    Fixed Fractional (default):
        size = (balance * risk_pct) / (entry - stop_loss)

    Kelly Criterion (optional):
        f* = (p * b - q) / b
        where p = win rate, b = avg_win/avg_loss, q = 1 - p
        Capped at max_risk_per_trade for safety.
    """

    def __init__(self, risk_config: RiskConfig):
        self.cfg = risk_config

    def calculate(
        self,
        signal: Signal,
        balance: float,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
    ) -> float:
        """
        Calculate the position size in base currency units.

        Args:
            signal: The trading signal (needs entry_price and stop_loss)
            balance: Available account balance in quote currency
            win_rate: Historical win rate (used for Kelly, optional)
            avg_win_loss_ratio: Average win / average loss ratio (optional)

        Returns:
            Position size in base currency units (e.g. BTC).
        """
        if signal.direction == SignalDirection.LONG:
            risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        else:
            risk_per_unit = abs(signal.stop_loss - signal.entry_price)

        if risk_per_unit <= 0:
            logger.warning("Invalid SL distance, using fallback sizing")
            risk_per_unit = signal.entry_price * 0.02

        # Determine risk fraction
        if win_rate is not None and avg_win_loss_ratio is not None:
            risk_fraction = self._kelly(win_rate, avg_win_loss_ratio)
        else:
            risk_fraction = self.cfg.max_risk_per_trade

        # Scale by signal confidence
        risk_fraction *= signal.confidence

        # Cap at configured maximum
        risk_fraction = min(risk_fraction, self.cfg.max_risk_per_trade)

        risk_amount = balance * risk_fraction
        size = risk_amount / risk_per_unit

        logger.debug(
            "Position size: {:.6f} (balance={:.2f}, risk_pct={:.2%}, entry={:.4f}, sl={:.4f})",
            size,
            balance,
            risk_fraction,
            signal.entry_price,
            signal.stop_loss,
        )
        return size

    def _kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Kelly fraction: f* = p - q/b
        p = win rate, q = loss rate, b = win/loss ratio
        """
        if win_loss_ratio <= 0:
            return self.cfg.max_risk_per_trade
        q = 1 - win_rate
        f = win_rate - q / win_loss_ratio
        # Half-Kelly for safety, capped
        return max(0.0, min(f * 0.5, self.cfg.max_risk_per_trade))
