from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trading_bot.data.models import MarketRegime, Position, TradeStatus
from trading_bot.monitoring.metrics import PerformanceMetrics
from trading_bot.data.models import Trade


class Dashboard:
    """
    Rich terminal dashboard showing live bot status.

    Displays:
      - Current market regime per symbol
      - Open positions with unrealised PnL
      - Closed trade performance summary
      - Latest signals
    """

    def __init__(self, refresh_interval: float = 2.0):
        self.refresh_interval = refresh_interval
        self._console = Console()
        self._regimes: Dict[str, MarketRegime] = {}
        self._open_positions: List[Position] = []
        self._closed_trades: List[Trade] = []
        self._latest_prices: Dict[str, float] = {}
        self._status_msg: str = "Running"
        self._started_at = datetime.utcnow()

    def update(
        self,
        regimes: Optional[Dict[str, MarketRegime]] = None,
        open_positions: Optional[List[Position]] = None,
        closed_trades: Optional[List[Trade]] = None,
        prices: Optional[Dict[str, float]] = None,
        status: Optional[str] = None,
    ) -> None:
        if regimes is not None:
            self._regimes = regimes
        if open_positions is not None:
            self._open_positions = open_positions
        if closed_trades is not None:
            self._closed_trades = closed_trades
        if prices is not None:
            self._latest_prices = prices
        if status is not None:
            self._status_msg = status

    def _regime_color(self, regime: MarketRegime) -> str:
        return {
            MarketRegime.TRENDING: "green",
            MarketRegime.RANGING: "yellow",
            MarketRegime.VOLATILE: "red",
            MarketRegime.BREAKOUT: "cyan",
        }.get(regime, "white")

    def _build_header(self) -> Panel:
        uptime = datetime.utcnow() - self._started_at
        text = Text()
        text.append("Adaptive Trading Bot", style="bold white")
        text.append(f"  |  Status: ", style="dim")
        color = "green" if "Running" in self._status_msg else "red"
        text.append(self._status_msg, style=f"bold {color}")
        text.append(f"  |  Uptime: {str(uptime).split('.')[0]}", style="dim")
        return Panel(text, style="bold blue")

    def _build_regime_table(self) -> Panel:
        table = Table(title="Market Regimes", expand=True)
        table.add_column("Symbol", style="bold")
        table.add_column("Regime")
        table.add_column("Price")
        for symbol, regime in self._regimes.items():
            price = self._latest_prices.get(symbol, 0.0)
            color = self._regime_color(regime)
            table.add_row(
                symbol,
                Text(regime.value.upper(), style=color),
                f"{price:,.4f}",
            )
        return Panel(table, title="[bold]Regimes[/bold]")

    def _build_positions_table(self) -> Panel:
        table = Table(title="Open Positions", expand=True)
        table.add_column("Symbol")
        table.add_column("Direction")
        table.add_column("Entry")
        table.add_column("Current")
        table.add_column("SL")
        table.add_column("TP")
        table.add_column("Strategy")

        for pos in self._open_positions:
            if pos.status != TradeStatus.OPEN:
                continue
            current = self._latest_prices.get(pos.symbol, pos.entry_price)
            direction_color = "green" if pos.direction.value == "LONG" else "red"
            table.add_row(
                pos.symbol,
                Text(pos.direction.value, style=direction_color),
                f"{pos.entry_price:,.4f}",
                f"{current:,.4f}",
                f"{pos.stop_loss:,.4f}",
                f"{pos.take_profit:,.4f}",
                pos.strategy_name[:20],
            )
        return Panel(table, title="[bold]Open Positions[/bold]")

    def _build_metrics_panel(self) -> Panel:
        if not self._closed_trades:
            return Panel("No closed trades yet.", title="[bold]Performance[/bold]")
        metrics = PerformanceMetrics(self._closed_trades)
        s = metrics.summary()
        lines = [
            f"Trades: {s['total_trades']}  |  Win Rate: {s['win_rate']:.1%}",
            f"Total PnL: {s['total_pnl']:+.4f}  |  Profit Factor: {s['profit_factor']:.2f}",
            f"Sharpe: {s['sharpe_ratio']:.2f}  |  Sortino: {s['sortino_ratio']:.2f}",
            f"Max Drawdown: {s['max_drawdown']:.1%}",
        ]
        return Panel("\n".join(lines), title="[bold]Performance[/bold]")

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._build_header(), size=3),
            Layout(name="main"),
        )
        layout["main"].split_row(
            Layout(self._build_regime_table()),
            Layout(name="right"),
        )
        layout["right"].split_column(
            Layout(self._build_positions_table()),
            Layout(self._build_metrics_panel()),
        )
        return layout

    def render_once(self) -> None:
        self._console.print(self._build_layout())

    def start_live(self) -> Live:
        """Return a Rich Live context manager. Use as: `with dashboard.start_live(): ...`"""
        return Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=1.0 / self.refresh_interval,
        )
