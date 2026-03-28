from __future__ import annotations

import math
from typing import List

import numpy as np

from trading_bot.data.models import Trade


class PerformanceMetrics:
    """
    Calculates trading performance statistics from a list of closed trades.
    """

    def __init__(self, trades: List[Trade]):
        self.trades = trades

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in sorted(self.trades, key=lambda x: x.closed_at):
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualised Sharpe Ratio (assumes daily returns)."""
        if len(self.trades) < 2:
            return 0.0
        returns = np.array([t.pnl for t in self.trades])
        excess = returns - risk_free_rate
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * math.sqrt(252))

    @property
    def sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Sortino Ratio (penalises only downside volatility)."""
        if len(self.trades) < 2:
            return 0.0
        returns = np.array([t.pnl for t in self.trades])
        downside = returns[returns < risk_free_rate]
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        return float((np.mean(returns) - risk_free_rate) / downside_std * math.sqrt(252))

    @property
    def avg_win_loss_ratio(self) -> float:
        if self.avg_loss == 0:
            return float("inf")
        return abs(self.avg_win / self.avg_loss)

    def summary(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "profit_factor": round(self.profit_factor, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "avg_win_loss_ratio": round(self.avg_win_loss_ratio, 3),
        }
