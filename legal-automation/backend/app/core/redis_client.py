"""
Gemeinsamer Redis-Client.

Vorher erzeugte jeder Aufrufer (`_get_redis()` im Login, das DSGVO-Dashboard,
der Health-Check) per `from_url()` einen EIGENEN Connection-Pool und schloss
ihn meist nicht — unter Last liefen damit die Redis-Verbindungen voll.
Hier gibt es genau einen Pool pro Prozess, der beim Shutdown geschlossen wird.
"""
from __future__ import annotations

import redis.asyncio as aioredis

from app.core.config import get_settings

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Liefert den prozessweiten Redis-Client (lazy erzeugt, wiederverwendet)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            health_check_interval=30,
        )
    return _client


async def close_redis() -> None:
    """Schließt den Pool (Lifespan-Shutdown; in Tests zwischen den Läufen)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
