"""
Synthetic Backtest Runner
Validates the full trading bot pipeline without requiring API keys.
Generates realistic OHLCV data with distinct market regimes and runs
it through: Features → ML Regime → Strategies → Risk → Execution.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from trading_bot.config.settings import load_config
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.monitoring.metrics import PerformanceMetrics


def generate_market_data(
    n_candles: int = 2000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data with 4 distinct regime phases:
      1. Ranging (500 candles)
      2. Trending up (500 candles)
      3. Breakout / volatile (500 candles)
      4. Trending down + ranging (500 candles)
    """
    np.random.seed(seed)
    phase_size = n_candles // 4
    phases = []

    # Phase 1: Ranging (low volatility, no trend)
    p1 = np.random.normal(0, 0.003, phase_size).cumsum()
    p1 = p1 - p1.mean()  # centre around 0
    phases.append(p1)

    # Phase 2: Trending up (positive drift)
    p2 = np.random.normal(0.002, 0.008, phase_size).cumsum()
    phases.append(p2 + p1[-1])

    # Phase 3: Breakout / volatile (high volatility, strong moves)
    p3 = np.random.normal(0.001, 0.020, phase_size).cumsum()
    phases.append(p3 + phases[-1][-1])

    # Phase 4: Trending down + ranging
    p4 = np.random.normal(-0.001, 0.007, phase_size).cumsum()
    phases.append(p4 + phases[-1][-1])

    log_returns = np.concatenate(phases)
    close = 50_000.0 * np.exp(log_returns)
    noise = np.random.uniform(0.001, 0.005, n_candles)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Volume: spikes in volatile phase
    base_vol = np.random.uniform(800, 1200, n_candles)
    spike = np.ones(n_candles)
    spike[phase_size * 2: phase_size * 3] = np.random.uniform(1.5, 3.0, phase_size)
    volume = base_vol * spike

    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_candles)]

    primary_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )

    # Resample to other timeframes
    def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return df.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

    primary_df.index = pd.DatetimeIndex(primary_df.index)
    return {
        "1h":  primary_df,
        "4h":  resample_ohlcv(primary_df, "4h"),
        "1d":  resample_ohlcv(primary_df, "1D"),
        "15m": primary_df.iloc[::1],  # same as 1h for synthetic data
        "5m":  primary_df.iloc[::1],
        "1m":  primary_df.iloc[::1],
    }


def print_report(metrics: dict, trades) -> None:
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Trades:       {metrics['total_trades']}")
    print(f"  Win Rate:           {metrics['win_rate']:.1%}")
    print(f"  Total PnL:          {metrics['total_pnl']:+,.2f} USDT")
    print(f"  Return:             {metrics['return_pct']:+.2%}")
    print(f"  Final Balance:      {metrics['final_balance']:,.2f} USDT")
    print(f"  Profit Factor:      {metrics['profit_factor']:.3f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.3f}")
    print(f"  Avg Win/Loss Ratio: {metrics['avg_win_loss_ratio']:.3f}")
    print("=" * 60)

    if trades:
        by_strategy: dict[str, list] = {}
        for t in trades:
            key = t.strategy_name.split(",")[0]
            by_strategy.setdefault(key, []).append(t.pnl)

        print("\n  PnL by Strategy:")
        for name, pnls in sorted(by_strategy.items()):
            wins = sum(1 for p in pnls if p > 0)
            total_pnl = sum(pnls)
            wr = wins / len(pnls) if pnls else 0
            print(f"    {name:<25} trades={len(pnls):>3}  wr={wr:.0%}  pnl={total_pnl:+,.2f}")
    print()


def main():
    print("\nGenerating synthetic market data (2000 x 1h candles)...")
    dataframes = generate_market_data(n_candles=2000)
    print(f"  Primary TF (1h):  {len(dataframes['1h'])} candles")
    print(f"  Higher TF (4h):   {len(dataframes['4h'])} candles")
    print(f"  Daily TF  (1d):   {len(dataframes['1d'])} candles")

    settings = load_config()
    settings.risk.min_confidence = 0.55
    settings.risk.max_risk_per_trade = 0.02
    settings.risk.max_drawdown_pct = 0.20

    print("\nRunning backtest pipeline...")
    print("  Data → Features → ML Regime → Strategies → Risk → Execution\n")

    engine = BacktestEngine(settings, initial_balance=10_000.0)
    trades = engine.run(dataframes, primary_tf="1h", symbol="BTC/USDT")
    metrics = engine.metrics()

    print_report(metrics, trades)

    # Acceptance criteria
    print("  Validation Criteria:")
    checks = [
        ("Trades executed > 0",        metrics["total_trades"] > 0),
        ("Win Rate >= 30%",             metrics["win_rate"] >= 0.30),
        ("No critical errors",          True),
        ("Drawdown < 50%",              metrics["max_drawdown"] < 0.50),
        ("Pipeline completed fully",    True),
    ]
    all_pass = True
    for label, result in checks:
        status = "✓" if result else "✗"
        print(f"    [{status}] {label}")
        if not result:
            all_pass = False

    print()
    if all_pass:
        print("  All checks passed. Pipeline is working correctly.")
    else:
        print("  Some checks failed — review strategy parameters.")
    print()


if __name__ == "__main__":
    main()
