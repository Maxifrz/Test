from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.core.audit import AuditMiddleware
from app.core.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run Alembic migrations on startup in non-test environments
    if settings.ENVIRONMENT != "test":
        import subprocess
        subprocess.run(["alembic", "upgrade", "head"], check=True)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    docs_url="/api/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/api/redoc" if settings.ENVIRONMENT == "development" else None,
    openapi_url="/api/openapi.json" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan,
)

# --- Middleware (order matters: outermost = last added) ---

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS + ["backend", "localhost"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"https://{h}" for h in settings.ALLOWED_HOSTS],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

app.add_middleware(AuditMiddleware)

# --- Routers ---

from app.api.routes import (
    auth, clients, matters, emails, tickets, calendar, transcription, finance,
)

app.include_router(auth.router, prefix="/api")
app.include_router(clients.router, prefix="/api")
app.include_router(matters.router, prefix="/api")
app.include_router(emails.router, prefix="/api")
app.include_router(tickets.router, prefix="/api")
app.include_router(calendar.router, prefix="/api")
app.include_router(transcription.router, prefix="/api")
app.include_router(finance.router, prefix="/api")


# --- Health check (not audit-logged, excluded in AuditMiddleware) ---

@app.get("/api/health")
async def health():
    from sqlalchemy import text
    from app.core.deps import AsyncSessionLocal
    import redis.asyncio as aioredis

    db_ok = False
    redis_ok = False

    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    try:
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if (db_ok and redis_ok) else "degraded",
        "db": "connected" if db_ok else "error",
        "redis": "connected" if redis_ok else "error",
        "environment": settings.ENVIRONMENT,
    }
