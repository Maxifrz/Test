from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from trading_bot.data.models import OHLCV, Trade


class Storage:
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    timeframe   TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    open        REAL NOT NULL,
                    high        REAL NOT NULL,
                    low         REAL NOT NULL,
                    close       REAL NOT NULL,
                    volume      REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf
                    ON ohlcv(symbol, timeframe, timestamp);

                CREATE TABLE IF NOT EXISTS trades (
                    position_id  TEXT PRIMARY KEY,
                    symbol       TEXT NOT NULL,
                    direction    TEXT NOT NULL,
                    market_type  TEXT NOT NULL,
                    entry_price  REAL NOT NULL,
                    exit_price   REAL NOT NULL,
                    quantity     REAL NOT NULL,
                    leverage     INTEGER NOT NULL DEFAULT 1,
                    pnl          REAL NOT NULL,
                    strategy     TEXT NOT NULL,
                    regime       TEXT NOT NULL DEFAULT '',
                    opened_at    TEXT NOT NULL,
                    closed_at    TEXT NOT NULL
                );
            """)
        logger.debug("Database initialised at {}", self.db_path)

    def save_candles(self, candles: List[OHLCV]) -> None:
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO ohlcv
                   (symbol, timeframe, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        c.symbol,
                        c.timeframe,
                        c.timestamp.isoformat(),
                        c.open,
                        c.high,
                        c.low,
                        c.close,
                        c.volume,
                    )
                    for c in candles
                ],
            )

    def save_candle(self, candle: OHLCV) -> None:
        self.save_candles([candle])

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: Optional[datetime] = None,
    ) -> List[OHLCV]:
        query = "SELECT * FROM ohlcv WHERE symbol=? AND timeframe=?"
        params: list = [symbol, timeframe]
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        candles = [
            OHLCV(
                timestamp=datetime.fromisoformat(r["timestamp"]),
                symbol=r["symbol"],
                timeframe=r["timeframe"],
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["volume"],
            )
            for r in reversed(rows)
        ]
        return candles

    def save_trade(self, trade: Trade) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trades
                   (position_id, symbol, direction, market_type, entry_price,
                    exit_price, quantity, leverage, pnl, strategy, regime,
                    opened_at, closed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.position_id,
                    trade.symbol,
                    trade.direction,
                    trade.market_type,
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.leverage,
                    trade.pnl,
                    trade.strategy_name,
                    trade.regime,
                    trade.opened_at.isoformat(),
                    trade.closed_at.isoformat(),
                ),
            )

    def load_trades(self, symbol: Optional[str] = None, limit: int = 1000) -> List[Trade]:
        query = "SELECT * FROM trades"
        params: list = []
        if symbol:
            query += " WHERE symbol=?"
            params.append(symbol)
        query += " ORDER BY closed_at DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            Trade(
                position_id=r["position_id"],
                symbol=r["symbol"],
                direction=r["direction"],
                market_type=r["market_type"],
                entry_price=r["entry_price"],
                exit_price=r["exit_price"],
                quantity=r["quantity"],
                leverage=r["leverage"],
                pnl=r["pnl"],
                strategy_name=r["strategy"],
                regime=r["regime"],
                opened_at=datetime.fromisoformat(r["opened_at"]),
                closed_at=datetime.fromisoformat(r["closed_at"]),
            )
            for r in rows
        ]
