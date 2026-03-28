from __future__ import annotations

from typing import Optional

import ccxt
from loguru import logger

from trading_bot.config.settings import ExchangeConfig

_PLACEHOLDER_KEYS = {"", "PLACEHOLDER_SPOT_KEY", "PLACEHOLDER_SPOT_SECRET",
                     "PLACEHOLDER_FUTURES_KEY", "PLACEHOLDER_FUTURES_SECRET"}


class Broker:
    """
    Unified ccxt exchange abstraction for Spot and Futures markets.

    In paper / testnet mode with placeholder keys the exchange object is
    created but all network calls are skipped gracefully, so the bot can
    run a full paper-trading session without real credentials.
    """

    def __init__(self, exchange_config: ExchangeConfig, market_type: str = "spot"):
        self.cfg = exchange_config
        self.market_type = market_type
        self._paper_mode = self._detect_paper_mode()
        self._exchange: Optional[ccxt.Exchange] = None
        if not self._paper_mode:
            self._exchange = self._build_exchange()

    def _detect_paper_mode(self) -> bool:
        """Paper mode if keys are missing/placeholder or testnet is forced."""
        is_futures = self.market_type == "futures"
        key = self.cfg.futures_api_key if is_futures else self.cfg.spot_api_key
        secret = self.cfg.futures_api_secret if is_futures else self.cfg.spot_api_secret
        if key in _PLACEHOLDER_KEYS or secret in _PLACEHOLDER_KEYS:
            logger.info(
                "No real API keys found for {} — running in paper mode (no exchange connection).",
                self.market_type,
            )
            return True
        return False

    def _build_exchange(self) -> ccxt.Exchange:
        is_futures = self.market_type == "futures"
        api_key = self.cfg.futures_api_key if is_futures else self.cfg.spot_api_key
        api_secret = self.cfg.futures_api_secret if is_futures else self.cfg.spot_api_secret

        options: dict = {}
        if is_futures:
            options["defaultType"] = "future"

        params = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": options,
        }

        exchange_class = getattr(ccxt, self.cfg.name)
        exchange = exchange_class(params)

        if self.cfg.testnet and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)
            logger.info("Exchange {} running in TESTNET / sandbox mode", self.cfg.name)

        return exchange

    def get_balance(self, currency: str = "USDT") -> float:
        if self._paper_mode or self._exchange is None:
            return 10_000.0  # simulated paper balance
        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get("free", {}).get(currency, 0.0))
        except Exception as exc:
            logger.error("Failed to fetch balance: {}", exc)
            return 0.0

    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[dict]:
        if self._paper_mode or self._exchange is None:
            return None  # paper executor handles this
        try:
            order = self._exchange.create_limit_order(symbol, side, amount, price)
            logger.info("Limit order placed: {} {} {} @ {}", side, amount, symbol, price)
            return order
        except Exception as exc:
            logger.error("Limit order failed: {}", exc)
            return None

    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[dict]:
        if self._paper_mode or self._exchange is None:
            return None
        try:
            order = self._exchange.create_market_order(symbol, side, amount)
            logger.info("Market order placed: {} {} {}", side, amount, symbol)
            return order
        except Exception as exc:
            logger.error("Market order failed: {}", exc)
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self._paper_mode or self._exchange is None:
            return True
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            logger.error("Cancel order failed: {}", exc)
            return False

    def fetch_order(self, order_id: str, symbol: str) -> Optional[dict]:
        if self._paper_mode or self._exchange is None:
            return None
        try:
            return self._exchange.fetch_order(order_id, symbol)
        except Exception as exc:
            logger.error("Fetch order failed: {}", exc)
            return None

    def fetch_ticker(self, symbol: str) -> Optional[dict]:
        if self._paper_mode or self._exchange is None:
            return None
        try:
            return self._exchange.fetch_ticker(symbol)
        except Exception as exc:
            logger.error("Fetch ticker failed: {}", exc)
            return None

    @property
    def is_paper(self) -> bool:
        return self._paper_mode or self.cfg.testnet
