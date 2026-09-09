from datetime import UTC, datetime
from typing import Any
import asyncio
import json
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings

logger = logging.getLogger("app.audit")


# Mapping HTTP method + path pattern → semantic action name
_METHOD_ACTION_MAP = {
    "GET": "read",
    "POST": "create",
    "PUT": "update",
    "PATCH": "update",
    "DELETE": "delete",
}

# Paths that are too noisy to log (health checks, metrics, static assets)
_SKIP_PATHS = {
    "/api/health",
    "/api/health/",
    "/api/health/live",
    "/api/metrics",
    "/favicon.ico",
}

# Laufende Audit-Schreibvorgänge. asyncio hält KEINE starke Referenz auf Tasks
# aus create_task() — ohne dieses Set kann der Garbage Collector einen Task
# einsammeln, bevor er ausgeführt wurde, und der Eintrag verschwindet still.
# Für ein Audit-Log, das per DB-Trigger unveränderlich gemacht wird, ist genau
# das der gefährlichere Fehlermodus.
_pending_writes: set[asyncio.Task] = set()


def _parse_resource(method: str, path: str) -> tuple[str, str]:
    """Extract (action, resource) from HTTP method and URL path."""
    action = _METHOD_ACTION_MAP.get(method.upper(), method.lower())
    parts = [p for p in path.strip("/").split("/") if p]
    # e.g. /api/matters/42/tickets → resource = "matter:42" for matter-scoped routes
    if len(parts) >= 3 and parts[0] == "api":
        resource_type = parts[1]
        resource_id = parts[2] if len(parts) > 2 and parts[2].isdigit() else None
        sub = parts[3] if len(parts) > 3 else None
        resource = f"{resource_type}:{resource_id}" if resource_id else resource_type
        if sub:
            action = f"{resource_type}.{sub}.{action}"
        else:
            action = f"{resource_type}.{action}"
        return action, resource
    return action, path


def _real_ip(client_host: str, forwarded_for: str | None) -> str:
    """
    Ermittelt die Client-IP. X-Forwarded-For wird nur akzeptiert, wenn die
    Verbindung aus dem konfigurierten Proxy-Netz kommt — sonst könnte jeder
    Client seine IP im Audit-Log frei wählen.
    """
    settings = get_settings()
    for prefix in settings.trusted_proxy_prefixes:
        if client_host.startswith(prefix):
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()
            break
    return client_host


async def write_audit_event(
    db: AsyncSession,
    *,
    user_id: int | None,
    user_email: str,
    session_id: str,
    ip_address: str,
    user_agent: str | None,
    action: str,
    resource: str,
    matter_id: int | None = None,
    client_id: int | None = None,
    details: dict[str, Any] | None = None,
    result: str,
) -> None:
    """Write an audit log entry. The app DB user has INSERT-only access to this table."""
    from sqlalchemy import text
    await db.execute(
        text(
            """
            INSERT INTO audit_log (
                user_id, user_email, session_id, ip_address, user_agent,
                action, resource, matter_id, client_id, details, result
            ) VALUES (
                :user_id, :user_email, :session_id, :ip_address, :user_agent,
                :action, :resource, :matter_id, :client_id, :details::jsonb, :result
            )
            """
        ),
        {
            "user_id": user_id,
            "user_email": user_email,
            "session_id": session_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "action": action,
            "resource": resource,
            "matter_id": matter_id,
            "client_id": client_id,
            "details": json.dumps(details) if details else None,
            "result": result,
        },
    )
    await db.commit()


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every API request to the audit log after the response is sent."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _SKIP_PATHS or not request.url.path.startswith("/api/"):
            return await call_next(request)

        # Request-Daten VOR call_next einsammeln: der Request-Scope kann nach
        # dem Senden der Antwort bereits abgeräumt sein.
        method = request.method
        path = request.url.path
        client_host = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        user_agent = request.headers.get("User-Agent")

        response = await call_next(request)

        user = getattr(request.state, "user", None)
        session_id = getattr(request.state, "session_id", "anonymous")
        action, resource = _parse_resource(method, path)
        result = (
            "success" if response.status_code < 400
            else ("denied" if response.status_code in (401, 403) else "error")
        )

        payload = {
            "user_id": user.id if user else None,
            "user_email": user.email if user else "anonymous",
            "session_id": str(session_id),
            "ip_address": _real_ip(client_host, forwarded_for),
            "user_agent": user_agent,
            "action": action,
            "resource": resource,
            "result": result,
        }

        task = asyncio.create_task(_write_audit_safely(payload))
        # Starke Referenz halten, bis der Task fertig ist (siehe _pending_writes).
        _pending_writes.add(task)
        task.add_done_callback(_pending_writes.discard)

        return response


async def _write_audit_safely(payload: dict[str, Any]) -> None:
    """
    Schreibt einen Audit-Eintrag und protokolliert Fehler laut. Ein stiller
    Verlust wäre DSGVO-relevant (Art. 5 Abs. 2 Rechenschaftspflicht) — deshalb
    landet jeder Fehlschlag mindestens im Anwendungslog.
    """
    from app.core.deps import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            await write_audit_event(db, **payload)
    except Exception:
        logger.exception(
            "Audit-Eintrag konnte nicht geschrieben werden: %s auf %s durch %s",
            payload.get("action"), payload.get("resource"), payload.get("user_email"),
        )


async def flush_pending_audit_writes(timeout: float = 5.0) -> int:
    """
    Wartet beim Shutdown auf offene Audit-Schreibvorgänge. Gibt die Anzahl der
    Tasks zurück, die nicht mehr rechtzeitig fertig wurden.
    """
    if not _pending_writes:
        return 0
    pending = set(_pending_writes)
    done, not_done = await asyncio.wait(pending, timeout=timeout)
    if not_done:
        logger.warning("%d Audit-Eintraege beim Shutdown nicht geschrieben", len(not_done))
    return len(not_done)
