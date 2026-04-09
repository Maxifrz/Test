from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import ccxt
from loguru import logger

from trading_bot.config.settings import Settings
from trading_bot.data.models import OHLCV
from trading_bot.data.storage import Storage


class HistoricalDataFetcher:
    """Fetches historical OHLCV data via ccxt REST API."""

    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage
        self._exchange = self._build_exchange()

    def _build_exchange(self) -> ccxt.Exchange:
        cfg = self.settings.exchange
        params = {
            "apiKey": cfg.spot_api_key,
            "secret": cfg.spot_api_secret,
            "enableRateLimit": True,
        }
        if cfg.testnet:
            params["options"] = {"defaultType": "spot"}
        exchange_class = getattr(ccxt, cfg.name)
        exchange = exchange_class(params)
        if cfg.testnet and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)
        return exchange

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: Optional[datetime] = None,
    ) -> List[OHLCV]:
        since_ms: Optional[int] = None
        if since:
            since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)

        logger.info("Fetching {} {} candles for {}", limit, timeframe, symbol)
        try:
            raw = self._exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except Exception as exc:
            logger.error("Failed to fetch OHLCV: {}", exc)
            return []

        candles = [OHLCV.from_ccxt(row, symbol, timeframe) for row in raw]
        if candles:
            self.storage.save_candles(candles)
        return candles

    def fetch_all_timeframes(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        limit: int = 500,
    ) -> Dict[str, List[OHLCV]]:
        if timeframes is None:
            timeframes = self.settings.trading.timeframes
        result: Dict[str, List[OHLCV]] = {}
        for tf in timeframes:
            result[tf] = self.fetch(symbol, tf, limit=limit)
        return result

    def backfill(self, symbol: str, timeframe: str, since: datetime) -> List[OHLCV]:
        """Backfill OHLCV from a start date, paging through the API."""
        all_candles: List[OHLCV] = []
        since_dt = since
        limit = 1000

        while True:
            batch = self.fetch(symbol, timeframe, limit=limit, since=since_dt)
            if not batch:
                break
            all_candles.extend(batch)
            last_ts = batch[-1].timestamp
            if last_ts <= since_dt or len(batch) < limit:
                break
            since_dt = last_ts

        logger.info(
            "Backfilled {} candles for {} {} since {}",
            len(all_candles),
            symbol,
            timeframe,
            since,
        )
        return all_candles
