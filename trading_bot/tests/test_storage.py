"""Tests for SQLite storage layer."""
from __future__ import annotations

from datetime import datetime

import pytest

from trading_bot.data.models import OHLCV, Trade
from trading_bot.data.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(db_path=str(tmp_path / "test.db"))


def make_candle(symbol="BTC/USDT", tf="1h", ts=None):
    return OHLCV(
        timestamp=ts or datetime(2024, 1, 1, 12, 0),
        symbol=symbol,
        timeframe=tf,
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        volume=1500.0,
    )


class TestStorage:
    def test_save_and_load_candle(self, storage):
        candle = make_candle()
        storage.save_candle(candle)
        candles = storage.load_candles("BTC/USDT", "1h")
        assert len(candles) == 1
        assert candles[0].close == pytest.approx(103.0)

    def test_duplicate_candle_not_inserted(self, storage):
        candle = make_candle()
        storage.save_candle(candle)
        storage.save_candle(candle)  # duplicate
        candles = storage.load_candles("BTC/USDT", "1h")
        assert len(candles) == 1

    def test_save_multiple_candles(self, storage):
        candles = [
            make_candle(ts=datetime(2024, 1, 1, i, 0)) for i in range(10)
        ]
        storage.save_candles(candles)
        loaded = storage.load_candles("BTC/USDT", "1h", limit=20)
        assert len(loaded) == 10

    def test_save_and_load_trade(self, storage):
        trade = Trade(
            position_id="pos-1",
            symbol="BTC/USDT",
            direction="LONG",
            market_type="spot",
            entry_price=100.0,
            exit_price=110.0,
            quantity=0.5,
            leverage=1,
            pnl=5.0,
            strategy_name="trend_following",
            regime="trending",
            opened_at=datetime(2024, 1, 1),
            closed_at=datetime(2024, 1, 2),
        )
        storage.save_trade(trade)
        trades = storage.load_trades()
        assert len(trades) == 1
        assert trades[0].pnl == pytest.approx(5.0)
