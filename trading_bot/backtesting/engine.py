from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from trading_bot.config.settings import Settings
from trading_bot.data.models import (
    MarketRegime,
    Position,
    Signal,
    SignalDirection,
    Trade,
    TradeStatus,
)
from trading_bot.features.indicators import compute_indicators
from trading_bot.features.multi_tf import MultiTFFeatures
from trading_bot.features.regime import build_regime_features, heuristic_regime
from trading_bot.ml.regime_classifier import RegimeClassifier
from trading_bot.ml.strategy_selector import StrategySelector
from trading_bot.monitoring.metrics import PerformanceMetrics
from trading_bot.risk.portfolio import PortfolioRisk
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.risk.stop_loss import ATRStopLoss
from trading_bot.signals.aggregator import SignalAggregator
from trading_bot.strategies import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)


class BacktestEngine:
    """
    Event-driven backtesting engine that replays historical OHLCV data
    through the full bot pipeline.

    Usage:
        engine = BacktestEngine(settings, initial_balance=10_000)
        trades = engine.run(dataframes, primary_tf="1h")
        metrics = engine.metrics()
    """

    def __init__(self, settings: Settings, initial_balance: float = 10_000.0):
        self.settings = settings
        self.initial_balance = initial_balance
        self.balance = initial_balance

        cfg = settings.strategies
        self._strategies = {
            "trend_following": TrendFollowingStrategy(vars(cfg.trend)),
            "mean_reversion": MeanReversionStrategy(vars(cfg.mean_reversion)),
            "breakout": BreakoutStrategy(vars(cfg.breakout)),
            "momentum": MomentumStrategy(vars(cfg.momentum)),
        }

        self._classifier = RegimeClassifier(settings.ml.model_path + "regime_classifier.pkl")
        self._selector = StrategySelector()
        self._aggregator = SignalAggregator()
        self._portfolio = PortfolioRisk(settings.risk)
        self._sizer = PositionSizer(settings.risk)
        self._sl_calc = ATRStopLoss(settings.risk)
        self._multi_tf = MultiTFFeatures(settings.trading.timeframes)

        self._open_positions: Dict[str, Position] = {}
        self._closed_trades: List[Trade] = []

    def run(
        self,
        dataframes: Dict[str, pd.DataFrame],
        primary_tf: str = "1h",
        symbol: str = "BTC/USDT",
    ) -> List[Trade]:
        """
        Run backtest over provided DataFrames.

        Args:
            dataframes: {timeframe: OHLCV DataFrame}
            primary_tf: The main timeframe to iterate over
            symbol: Trading symbol

        Returns:
            List of completed Trade records
        """
        primary_df = dataframes.get(primary_tf)
        if primary_df is None or primary_df.empty:
            logger.error("No data for primary timeframe {}", primary_tf)
            return []

        # Enrich all timeframes
        for tf, df in dataframes.items():
            self._multi_tf.update(symbol, tf, df)

        logger.info(
            "Starting backtest: {} candles on {}, balance={:.2f}",
            len(primary_df),
            primary_tf,
            self.initial_balance,
        )

        enriched = compute_indicators(primary_df, vars(self.settings.strategies.trend))

        for i in range(50, len(enriched)):
            window = enriched.iloc[:i]
            current_price = float(window.iloc[-1]["close"])
            current_time = window.index[-1]

            # Update balance tracking
            self._portfolio.update_drawdown(self.balance)
            if self._portfolio.is_halted:
                break

            # Check exits for open positions
            self._check_exits(symbol, current_price, current_time)

            # Build features and detect regime
            features = build_regime_features(window)
            regime, _ = self._classifier.predict(features)

            # Run strategies
            multi_tf_frames = self._multi_tf.get(symbol)
            signals: Dict[str, Optional[Signal]] = {}
            for name, strategy in self._strategies.items():
                signals[name] = strategy.generate_signal(symbol, window, multi_tf_frames, regime)

            weights = self._selector.get_weights(regime)
            final_signal = self._aggregator.aggregate(
                signals, weights, self.settings.risk.min_confidence
            )

            if final_signal and self._portfolio.approve(final_signal, self.balance):
                qty = self._sizer.calculate(final_signal, self.balance)
                if qty > 0:
                    self._open_position(final_signal, qty, regime)

        # Force-close any remaining positions at last price
        last_price = float(enriched.iloc[-1]["close"])
        for pos in list(self._open_positions.values()):
            self._close_position(pos, last_price, enriched.index[-1])

        logger.info(
            "Backtest complete. Trades={}, Final balance={:.2f}, PnL={:.2f}",
            len(self._closed_trades),
            self.balance,
            self.balance - self.initial_balance,
        )
        return self._closed_trades

    def _open_position(self, signal: Signal, qty: float, regime: MarketRegime) -> None:
        pos = Position(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            direction=signal.direction,
            market_type=__import__("trading_bot.data.models", fromlist=["MarketType"]).MarketType.SPOT,
            entry_price=signal.entry_price,
            quantity=qty,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            leverage=1,
            opened_at=signal.timestamp,
            strategy_name=signal.strategy_name,
        )
        self._open_positions[pos.id] = pos
        self._portfolio.add_position(pos)

    def _check_exits(self, symbol: str, price: float, ts: datetime) -> None:
        for pos_id, pos in list(self._open_positions.items()):
            if pos.symbol != symbol:
                continue
            should_close = False
            if pos.direction == SignalDirection.LONG:
                should_close = price <= pos.stop_loss or price >= pos.take_profit
            else:
                should_close = price >= pos.stop_loss or price <= pos.take_profit
            if should_close:
                self._close_position(pos, price, ts)

    def _close_position(self, pos: Position, price: float, ts: datetime) -> None:
        pos.close(price)
        pos.closed_at = ts
        self.balance += pos.pnl
        self._portfolio.close_position(pos.id, price)
        trade = Trade(
            position_id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction.value,
            market_type=pos.market_type.value,
            entry_price=pos.entry_price,
            exit_price=pos.exit_price or price,
            quantity=pos.quantity,
            leverage=pos.leverage,
            pnl=pos.pnl,
            strategy_name=pos.strategy_name,
            opened_at=pos.opened_at,
            closed_at=pos.closed_at or ts,
        )
        self._closed_trades.append(trade)
        del self._open_positions[pos.id]

    def metrics(self) -> dict:
        m = PerformanceMetrics(self._closed_trades, initial_balance=self.initial_balance)
        result = m.summary()
        result["final_balance"] = round(self.balance, 4)
        result["return_pct"] = round(
            (self.balance - self.initial_balance) / self.initial_balance, 4
        )
        return result
