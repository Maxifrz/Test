"""Tests for signal aggregation."""
from __future__ import annotations

from trading_bot.data.models import MarketRegime, Signal, SignalDirection
from trading_bot.signals.aggregator import SignalAggregator


def make_signal(direction=SignalDirection.LONG, confidence=0.7, strategy="trend_following"):
    return Signal(
        symbol="BTC/USDT",
        direction=direction,
        confidence=confidence,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        strategy_name=strategy,
        regime=MarketRegime.TRENDING,
        timeframe="1h",
    )


class TestSignalAggregator:
    def test_all_none_returns_none(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            {"trend_following": None, "momentum": None},
            {"trend_following": 0.6, "momentum": 0.4},
        )
        assert result is None

    def test_single_signal_passes(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            {"trend_following": make_signal(confidence=0.75), "momentum": None},
            {"trend_following": 0.8, "momentum": 0.2},
            min_confidence=0.5,
        )
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_conflicting_signals_go_flat(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            {
                "trend_following": make_signal(SignalDirection.LONG, 0.65),
                "mean_reversion": make_signal(SignalDirection.SHORT, 0.65),
            },
            {"trend_following": 0.5, "mean_reversion": 0.5},
            min_confidence=0.6,
        )
        # With equal weights and opposite directions, weighted score ~ 0.325 each → below 0.6
        assert result is None

    def test_dominant_direction_wins(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            {
                "trend_following": make_signal(SignalDirection.LONG, 0.80),
                "momentum": make_signal(SignalDirection.LONG, 0.70),
                "mean_reversion": make_signal(SignalDirection.SHORT, 0.60),
            },
            {"trend_following": 0.5, "momentum": 0.3, "mean_reversion": 0.2},
            min_confidence=0.55,
        )
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_aggregated_signal_is_valid(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            {"trend_following": make_signal(confidence=0.80)},
            {"trend_following": 1.0},
            min_confidence=0.5,
        )
        assert result is not None
        assert result.is_valid
