from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from trading_bot.config.settings import Settings
from trading_bot.data.models import OHLCV
from trading_bot.data.storage import Storage


class LiveFeed:
    """Streams real-time OHLCV data via ccxt.pro WebSocket."""

    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage
        # in-memory candle buffer: symbol -> timeframe -> list[OHLCV]
        self._buffers: Dict[str, Dict[str, List[OHLCV]]] = defaultdict(lambda: defaultdict(list))
        self._callbacks: List[Callable] = []
        self._running = False
        self._exchange = None

    def register_callback(self, fn: Callable[[str, str, OHLCV], None]) -> None:
        """Register a function to be called on every new closed candle."""
        self._callbacks.append(fn)

    def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> List[OHLCV]:
        buf = self._buffers[symbol][timeframe]
        return buf[-limit:] if len(buf) >= limit else buf[:]

    def get_dataframe(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        candles = self.get_candles(symbol, timeframe, limit)
        if not candles:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]
        ).set_index("timestamp")

    async def _watch_symbol_tf(self, exchange, symbol: str, timeframe: str) -> None:
        while self._running:
            try:
                raw_candles = await exchange.watch_ohlcv(symbol, timeframe)
                for row in raw_candles:
                    candle = OHLCV.from_ccxt(row, symbol, timeframe)
                    buf = self._buffers[symbol][timeframe]
                    # replace last candle (live) or append closed candle
                    if buf and buf[-1].timestamp == candle.timestamp:
                        buf[-1] = candle
                    else:
                        buf.append(candle)
                        self.storage.save_candle(candle)
                        for cb in self._callbacks:
                            try:
                                cb(symbol, timeframe, candle)
                            except Exception as exc:
                                logger.error("Callback error: {}", exc)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("WebSocket error for {} {}: {}. Reconnecting…", symbol, timeframe, exc)
                await asyncio.sleep(5)

    async def start(self) -> None:
        """Start all WebSocket streams concurrently."""
        try:
            import ccxt.pro as ccxtpro
        except ImportError:
            logger.error("ccxt[pro] is not installed. Run: pip install 'ccxt[pro]'")
            return

        cfg = self.settings.exchange
        exchange = getattr(ccxtpro, cfg.name)(
            {
                "apiKey": cfg.spot_api_key,
                "secret": cfg.spot_api_secret,
                "enableRateLimit": True,
            }
        )
        if cfg.testnet and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)

        self._exchange = exchange
        self._running = True
        symbols = self.settings.trading.symbols
        timeframes = self.settings.trading.timeframes

        tasks = [
            asyncio.create_task(self._watch_symbol_tf(exchange, symbol, tf))
            for symbol in symbols
            for tf in timeframes
        ]
        logger.info("LiveFeed started for {} symbols × {} timeframes", len(symbols), len(timeframes))
        try:
            await asyncio.gather(*tasks)
        finally:
            await exchange.close()

    def stop(self) -> None:
        self._running = False
