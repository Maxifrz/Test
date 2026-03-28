"""Tests for risk management modules."""
from __future__ import annotations

from datetime import datetime

import pytest

from trading_bot.config.settings import RiskConfig
from trading_bot.data.models import MarketRegime, Position, Signal, SignalDirection, MarketType, TradeStatus
from trading_bot.risk.portfolio import PortfolioRisk
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.risk.stop_loss import ATRStopLoss


def make_signal(direction=SignalDirection.LONG, confidence=0.75, entry=100.0, sl=95.0, tp=115.0):
    return Signal(
        symbol="BTC/USDT",
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        strategy_name="test",
        regime=MarketRegime.TRENDING,
        timeframe="1h",
    )


def make_position(symbol="BTC/USDT", direction=SignalDirection.LONG):
    return Position(
        id="test-pos-1",
        symbol=symbol,
        direction=direction,
        market_type=MarketType.SPOT,
        entry_price=100.0,
        quantity=0.1,
        stop_loss=95.0,
        take_profit=115.0,
    )


class TestPositionSizer:
    def test_basic_sizing(self):
        cfg = RiskConfig(max_risk_per_trade=0.02)
        sizer = PositionSizer(cfg)
        signal = make_signal(entry=100.0, sl=95.0)
        qty = sizer.calculate(signal, balance=10_000)
        # risk = 10000 * 0.02 * 0.75 confidence = 150 / (100-95) = 30 units
        assert qty > 0
        assert qty == pytest.approx(10_000 * 0.02 * 0.75 / 5.0, rel=0.01)

    def test_kelly_sizing(self):
        cfg = RiskConfig(max_risk_per_trade=0.05)
        sizer = PositionSizer(cfg)
        signal = make_signal()
        qty = sizer.calculate(signal, balance=10_000, win_rate=0.6, avg_win_loss_ratio=1.5)
        assert qty > 0

    def test_invalid_sl_uses_fallback(self):
        cfg = RiskConfig()
        sizer = PositionSizer(cfg)
        signal = make_signal(entry=100.0, sl=100.0)  # zero distance
        qty = sizer.calculate(signal, balance=10_000)
        assert qty > 0  # fallback should still produce a size


class TestATRStopLoss:
    def test_long_sl_below_entry(self):
        cfg = RiskConfig(atr_sl_multiplier=2.0, atr_tp_multiplier=3.0)
        calc = ATRStopLoss(cfg)
        sl, tp = calc.calculate(SignalDirection.LONG, entry=100.0, atr=1.0)
        assert sl < 100.0
        assert tp > 100.0
        assert sl == pytest.approx(98.0)
        assert tp == pytest.approx(103.0)

    def test_short_sl_above_entry(self):
        cfg = RiskConfig(atr_sl_multiplier=2.0, atr_tp_multiplier=3.0)
        calc = ATRStopLoss(cfg)
        sl, tp = calc.calculate(SignalDirection.SHORT, entry=100.0, atr=1.0)
        assert sl > 100.0
        assert tp < 100.0

    def test_trailing_stop_long_only_moves_up(self):
        cfg = RiskConfig()
        calc = ATRStopLoss(cfg)
        new_sl = calc.trailing_stop(SignalDirection.LONG, current_price=110.0, current_sl=95.0, atr=2.0)
        assert new_sl > 95.0
        # Should not move down
        lower_sl = calc.trailing_stop(SignalDirection.LONG, current_price=100.0, current_sl=new_sl, atr=2.0)
        assert lower_sl >= new_sl


class TestPortfolioRisk:
    def test_approve_basic(self):
        cfg = RiskConfig(max_open_positions=3, max_portfolio_exposure=0.5)
        portfolio = PortfolioRisk(cfg)
        signal = make_signal()
        assert portfolio.approve(signal, balance=10_000) is True

    def test_reject_when_max_positions_reached(self):
        cfg = RiskConfig(max_open_positions=1)
        portfolio = PortfolioRisk(cfg)
        pos = make_position()
        portfolio.add_position(pos)
        signal = make_signal(direction=SignalDirection.SHORT)
        assert portfolio.approve(signal, balance=10_000) is False

    def test_reject_duplicate_symbol_direction(self):
        cfg = RiskConfig(max_open_positions=5)
        portfolio = PortfolioRisk(cfg)
        pos = make_position()
        portfolio.add_position(pos)
        signal = make_signal()  # same symbol + direction
        assert portfolio.approve(signal, balance=10_000) is False

    def test_circuit_breaker_halts_trading(self):
        cfg = RiskConfig(max_drawdown_pct=0.10)
        portfolio = PortfolioRisk(cfg)
        portfolio.update_drawdown(10_000)  # set peak
        portfolio.update_drawdown(8_000)   # 20% drawdown → halted
        assert portfolio.is_halted is True
        signal = make_signal()
        assert portfolio.approve(signal, balance=8_000) is False

    def test_close_position(self):
        cfg = RiskConfig()
        portfolio = PortfolioRisk(cfg)
        pos = make_position()
        portfolio.add_position(pos)
        closed = portfolio.close_position(pos.id, exit_price=110.0)
        assert closed is not None
        assert closed.pnl == pytest.approx(1.0)  # (110-100)*0.1
        assert closed.status == TradeStatus.CLOSED
