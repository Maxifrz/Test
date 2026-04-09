#!/usr/bin/env python3
"""
Adaptive Multi-Strategy Crypto Trading Bot
Entry point: asyncio event loop that orchestrates all modules.

Usage:
  python main.py                        # paper trading (default)
  python main.py --mode paper           # explicit paper mode
  python main.py --mode live            # live trading (requires API keys)
  python main.py --mode backtest        # run backtest on historical data
  python main.py --config my_config.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from trading_bot.config.settings import load_config, Settings
from trading_bot.data.feed import LiveFeed
from trading_bot.data.historical import HistoricalDataFetcher
from trading_bot.data.models import MarketRegime, Trade
from trading_bot.data.storage import Storage
from trading_bot.execution.broker import Broker
from trading_bot.execution.futures_executor import FuturesExecutor
from trading_bot.execution.spot_executor import SpotExecutor
from trading_bot.features.multi_tf import MultiTFFeatures
from trading_bot.features.regime import build_regime_features
from trading_bot.ml.regime_classifier import RegimeClassifier
from trading_bot.ml.strategy_selector import StrategySelector
from trading_bot.ml.trainer import Trainer
from trading_bot.monitoring.dashboard import Dashboard
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


def configure_logging(settings: Settings) -> None:
    log_path = Path(settings.monitoring.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=settings.monitoring.log_level, colorize=True)
    logger.add(log_path, rotation="10 MB", retention="7 days", level="DEBUG")


class TradingBot:
    """Main bot orchestrator."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = Storage()
        self.feed = LiveFeed(settings, self.storage)

        # Feature pipeline
        ind_cfg = vars(settings.strategies.trend)
        self.multi_tf = MultiTFFeatures(settings.trading.timeframes, ind_cfg)

        # ML
        self.classifier = RegimeClassifier(settings.ml.model_path + "regime_classifier.pkl")
        self.selector = StrategySelector()
        self.trainer = Trainer(self.storage, self.classifier, settings.ml.min_training_samples)

        # Strategies
        self.strategies = {
            "trend_following": TrendFollowingStrategy(vars(settings.strategies.trend)),
            "mean_reversion": MeanReversionStrategy(vars(settings.strategies.mean_reversion)),
            "breakout": BreakoutStrategy(vars(settings.strategies.breakout)),
            "momentum": MomentumStrategy(vars(settings.strategies.momentum)),
        }

        # Signal & Risk
        self.aggregator = SignalAggregator()
        self.portfolio = PortfolioRisk(settings.risk)
        self.sizer = PositionSizer(settings.risk)
        self.sl_calc = ATRStopLoss(settings.risk)

        # Execution
        mode = settings.trading.mode
        spot_broker = Broker(settings.exchange, market_type="spot")
        futures_broker = Broker(settings.exchange, market_type="futures")
        self.spot_executor = SpotExecutor(spot_broker, mode=mode)
        self.futures_executor = FuturesExecutor(futures_broker, settings.risk, mode=mode)

        # Dashboard
        self.dashboard = Dashboard(settings.monitoring.refresh_interval)
        self._regimes: Dict[str, MarketRegime] = {}
        self._latest_prices: Dict[str, float] = {}

    def _on_new_candle(self, symbol: str, timeframe: str, candle) -> None:
        """Callback: runs on every new closed candle from the live feed."""
        df = self.feed.get_dataframe(symbol, timeframe)
        if df.empty:
            return

        self.multi_tf.update(symbol, timeframe, df)
        self._latest_prices[symbol] = float(candle.close)

        # Only generate signals on the primary timeframe
        if timeframe != self.settings.trading.primary_tf:
            return

        if not self.multi_tf.is_ready(symbol):
            return

        self._process_symbol(symbol)

    def _process_symbol(self, symbol: str) -> None:
        primary_tf = self.settings.trading.primary_tf
        primary_df = self.multi_tf.get(symbol).get(primary_tf)
        if primary_df is None or primary_df.empty:
            return

        # Update balance & drawdown check
        balance = self._get_balance()
        self.portfolio.update_drawdown(balance)
        if self.portfolio.is_halted:
            self.dashboard.update(status="HALTED: Max drawdown reached")
            return

        # Regime detection
        features = build_regime_features(primary_df, self.settings.ml.regime_lookback)
        regime, confidence = self.classifier.predict(features)
        self._regimes[symbol] = regime
        logger.debug("Regime {} for {}: {} ({:.0%})", symbol, primary_tf, regime.value, confidence)

        # Check open position exits
        for pos in list(self.portfolio.open_positions):
            if pos.symbol != symbol:
                continue
            current_price = self._latest_prices.get(symbol, pos.entry_price)
            should_exit = self.spot_executor.check_exit(pos, current_price)
            if should_exit:
                closed = self.portfolio.close_position(pos.id, current_price)
                if closed:
                    trade = self.spot_executor.build_trade_record(closed, regime=regime.value)
                    self.storage.save_trade(trade)

        # Generate signals from all strategies
        multi_tf_frames = self.multi_tf.get(symbol)
        signals = {}
        for name, strategy in self.strategies.items():
            signals[name] = strategy.generate_signal(symbol, primary_df, multi_tf_frames, regime)

        weights = self.selector.get_weights(regime)
        final_signal = self.aggregator.aggregate(
            signals, weights, self.settings.risk.min_confidence
        )

        if final_signal and self.portfolio.approve(final_signal, balance):
            qty = self.sizer.calculate(final_signal, balance)
            if qty > 0:
                asyncio.create_task(self._execute_signal(final_signal, qty))

        # Update dashboard
        self.dashboard.update(
            regimes=self._regimes.copy(),
            open_positions=self.portfolio.open_positions,
            closed_trades=self.storage.load_trades(limit=100),
            prices=self._latest_prices.copy(),
            status="Running" if not self.portfolio.is_halted else "HALTED",
        )

    async def _execute_signal(self, signal, qty: float) -> None:
        position = await self.spot_executor.execute(signal, qty)
        if position:
            self.portfolio.add_position(position)
            logger.info("Position opened: {} {} qty={:.6f}", signal.symbol, signal.direction.value, qty)

    def _get_balance(self) -> float:
        broker = self.spot_executor.broker
        if self.settings.trading.mode == "paper":
            # Estimate paper balance
            closed = self.storage.load_trades(limit=1000)
            pnl = sum(t.pnl for t in closed)
            initial = 10_000.0  # default paper balance
            return initial + pnl
        return broker.get_balance("USDT")

    async def run(self) -> None:
        """Main bot loop."""
        logger.info("Starting bot in {} mode", self.settings.trading.mode.upper())

        # Backfill historical data
        fetcher = HistoricalDataFetcher(self.settings, self.storage)
        for symbol in self.settings.trading.symbols:
            fetcher.fetch_all_timeframes(symbol, limit=self.settings.trading.candle_limit)

        # Try initial training
        primary_symbol = self.settings.trading.symbols[0]
        primary_tf = self.settings.trading.primary_tf
        self.trainer.train(primary_symbol, primary_tf, force=False)

        # Populate multi-tf from storage
        for symbol in self.settings.trading.symbols:
            for tf in self.settings.trading.timeframes:
                candles = self.storage.load_candles(symbol, tf, limit=500)
                if candles:
                    import pandas as pd
                    df = pd.DataFrame(
                        [{"timestamp": c.timestamp, "open": c.open, "high": c.high,
                          "low": c.low, "close": c.close, "volume": c.volume}
                         for c in candles]
                    ).set_index("timestamp")
                    self.multi_tf.update(symbol, tf, df)

        # Register live feed callback
        self.feed.register_callback(self._on_new_candle)

        # Start dashboard and live feed
        with self.dashboard.start_live():
            # Schedule periodic retraining
            asyncio.create_task(self._retrain_loop())
            # Start WebSocket feed (blocks until stopped)
            await self.feed.start()

    async def _retrain_loop(self) -> None:
        """Periodically retrain the regime classifier."""
        import asyncio as _asyncio
        interval = self.settings.ml.retrain_interval_hours * 3600
        primary_symbol = self.settings.trading.symbols[0]
        primary_tf = self.settings.trading.primary_tf
        while True:
            await _asyncio.sleep(interval)
            logger.info("Scheduled retraining triggered")
            self.trainer.train(primary_symbol, primary_tf, force=True)


def run_backtest(settings: Settings) -> None:
    from trading_bot.backtesting.engine import BacktestEngine
    from trading_bot.data.historical import HistoricalDataFetcher

    storage = Storage()
    fetcher = HistoricalDataFetcher(settings, storage)
    symbol = settings.trading.symbols[0]

    logger.info("Running backtest for {}", symbol)
    dataframes = {}
    for tf in settings.trading.timeframes:
        candles = fetcher.fetch(symbol, tf, limit=1000)
        if candles:
            import pandas as pd
            df = pd.DataFrame(
                [{"timestamp": c.timestamp, "open": c.open, "high": c.high,
                  "low": c.low, "close": c.close, "volume": c.volume}
                 for c in candles]
            ).set_index("timestamp")
            dataframes[tf] = df

    engine = BacktestEngine(settings, initial_balance=10_000.0)
    engine.run(dataframes, primary_tf=settings.trading.primary_tf, symbol=symbol)
    metrics = engine.metrics()

    logger.info("Backtest results:")
    for k, v in metrics.items():
        logger.info("  {}: {}", k, v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Multi-Strategy Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live", "backtest"], default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or Path(__file__).parent / "trading_bot/config/config.yaml"
    settings = load_config(config_path)

    if args.mode:
        settings.trading.mode = args.mode

    configure_logging(settings)

    logger.info("Bot starting  |  mode={}  |  symbols={}", settings.trading.mode, settings.trading.symbols)

    if settings.trading.mode == "backtest":
        run_backtest(settings)
    else:
        bot = TradingBot(settings)
        asyncio.run(bot.run())


if __name__ == "__main__":
    main()
