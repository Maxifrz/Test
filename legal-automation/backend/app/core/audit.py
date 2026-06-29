from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings


# Mapping HTTP method + path pattern → semantic action name
_METHOD_ACTION_MAP = {
    "GET": "read",
    "POST": "create",
    "PUT": "update",
    "PATCH": "update",
    "DELETE": "delete",
}

# Paths that are too noisy to log (health checks, static assets)
_SKIP_PATHS = {"/api/health", "/api/health/", "/favicon.ico"}


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


def _real_ip(request: Request) -> str:
    """Extract real client IP, trusting X-Forwarded-For only from nginx (172.20.0.0/16)."""
    client_host = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for and client_host.startswith("172.20."):
        return forwarded_for.split(",")[0].strip()
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
            "details": __import__("json").dumps(details) if details else None,
            "result": result,
        },
    )
    await db.commit()


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every API request to the audit log after the response is sent."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _SKIP_PATHS or not request.url.path.startswith("/api/"):
            return await call_next(request)

        response = await call_next(request)

        # Fire-and-forget audit write in a background task
        user = getattr(request.state, "user", None)
        session_id = getattr(request.state, "session_id", "anonymous")
        action, resource = _parse_resource(request.method, request.url.path)
        result = "success" if response.status_code < 400 else ("denied" if response.status_code == 403 else "error")

        from fastapi import BackgroundTasks
        from app.core.deps import get_db_session

        async def _write():
            async for db in get_db_session():
                await write_audit_event(
                    db,
                    user_id=user.id if user else None,
                    user_email=user.email if user else "anonymous",
                    session_id=str(session_id),
                    ip_address=_real_ip(request),
                    user_agent=request.headers.get("User-Agent"),
                    action=action,
                    resource=resource,
                    result=result,
                )

        import asyncio
        asyncio.create_task(_write())

        return response
