"""
Betriebsmetriken im Prometheus-Textformat.

Bewusst ohne prometheus_client-Abhaengigkeit: die Kennzahlen kommen aus
Datenbank und Redis, nicht aus Prozess-Countern, und werden je Abruf frisch
ermittelt. Ein zusaetzliches Paket haette hier nur Registry-Verwaltung
gebracht.

Zugriffsschutz zweistufig:
  1. nginx laesst /api/metrics nur aus dem internen Netz durch.
  2. Ist METRICS_TOKEN gesetzt, wird zusaetzlich ein Bearer-Token verlangt --
     fuer Aufbauten, in denen der Scraper ausserhalb des Compose-Netzes laeuft.
"""
from fastapi import APIRouter, Header, HTTPException, Response, status
from sqlalchemy import func, select, text

from app.core.config import get_settings
from app.core.deps import DB
from app.models.dsgvo import ErasureRequest
from app.models.email import EmailMessage
from app.models.ticket import Ticket
from app.models.user import User, UserSession

# Abgeschlossene Zustaende zaehlen nicht als ueberfaellig
TICKET_CLOSED_STATUSES = ("closed",)

router = APIRouter(tags=["metrics"])


def _line(name: str, value, help_text: str, metric_type: str = "gauge") -> str:
    return (
        f"# HELP {name} {help_text}\n"
        f"# TYPE {name} {metric_type}\n"
        f"{name} {value}\n"
    )


async def _celery_queue_depths() -> dict[str, int]:
    """
    Laenge der Celery-Warteschlangen. Ohne diese Kennzahl faellt ein
    haengender Worker erst auf, wenn jemand eine fehlende Transkription
    vermisst.
    """
    from app.core.redis_client import get_redis

    depths: dict[str, int] = {}
    try:
        redis = get_redis()
        for queue in ("transcription", "email", "default"):
            depths[queue] = int(await redis.llen(queue) or 0)
    except Exception:
        # Redis nicht erreichbar: keine Kennzahl statt einer falschen
        return {}
    return depths


@router.get("/metrics", response_class=Response)
async def metrics(db: DB, authorization: str | None = Header(default=None)):
    settings = get_settings()
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metriken deaktiviert")

    if settings.METRICS_TOKEN:
        expected = f"Bearer {settings.METRICS_TOKEN}"
        # Konstantzeit-Vergleich: der Endpunkt ist unauthentifiziert erreichbar
        import hmac

        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Ungueltiges Metrics-Token"
            )

    out: list[str] = []

    # --- Datenbank ---
    db_up = 1
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_up = 0
    out.append(_line("legal_db_up", db_up, "1 wenn die Datenbank erreichbar ist"))

    if db_up:
        active_sessions = (await db.execute(
            select(func.count()).select_from(UserSession).where(
                UserSession.is_revoked.is_(False), UserSession.expires_at > func.now()
            )
        )).scalar_one()
        out.append(_line("legal_active_sessions", active_sessions, "Offene Anmeldesitzungen"))

        users_total = (await db.execute(
            select(func.count()).select_from(User).where(User.deleted_at.is_(None))
        )).scalar_one()
        users_2fa = (await db.execute(
            select(func.count()).select_from(User).where(
                User.totp_enabled.is_(True), User.deleted_at.is_(None)
            )
        )).scalar_one()
        out.append(_line("legal_users_total", users_total, "Aktive Benutzerkonten"))
        out.append(_line("legal_users_with_2fa", users_2fa, "Benutzer mit eingerichteter 2FA"))

        # Die wichtigste Betriebskennzahl: haengengebliebene Ausgangsmails.
        # Vorher gab es dafuer ueberhaupt keine Sichtbarkeit.
        queued = (await db.execute(
            select(func.count()).select_from(EmailMessage).where(
                EmailMessage.direction == "outbound",
                EmailMessage.delivery_status == "queued",
            )
        )).scalar_one()
        failed = (await db.execute(
            select(func.count()).select_from(EmailMessage).where(
                EmailMessage.direction == "outbound",
                EmailMessage.delivery_status == "failed",
            )
        )).scalar_one()
        out.append(_line("legal_email_outbox_queued", queued, "Ausgangsmails in der Warteschlange"))
        out.append(_line(
            "legal_email_outbox_failed", failed,
            "Ausgangsmails, die endgueltig gescheitert sind (Handlungsbedarf)",
        ))

        overdue = (await db.execute(
            select(func.count()).select_from(Ticket).where(
                # due_date ist ein DATE -- current_date statt now() vermeidet
                # einen impliziten Timestamp-Vergleich
                Ticket.due_date < func.current_date(),
                Ticket.status.notin_(TICKET_CLOSED_STATUSES),
            )
        )).scalar_one()
        out.append(_line("legal_tickets_overdue", overdue, "Ueberfaellige Aufgaben/Fristen"))

        blocked = (await db.execute(
            select(func.count()).select_from(ErasureRequest).where(ErasureRequest.status == "blocked")
        )).scalar_one()
        open_erasure = (await db.execute(
            select(func.count()).select_from(ErasureRequest).where(ErasureRequest.status == "open")
        )).scalar_one()
        out.append(_line("legal_erasure_requests_open", open_erasure, "Offene Loeschantraege"))
        out.append(_line("legal_erasure_requests_blocked", blocked, "Blockierte Loeschantraege"))

    # --- Redis / Celery ---
    depths = await _celery_queue_depths()
    out.append(_line("legal_redis_up", 1 if depths else 0, "1 wenn Redis erreichbar ist"))
    if depths:
        out.append("# HELP legal_celery_queue_depth Anzahl wartender Celery-Tasks\n")
        out.append("# TYPE legal_celery_queue_depth gauge\n")
        for queue, depth in sorted(depths.items()):
            out.append(f'legal_celery_queue_depth{{queue="{queue}"}} {depth}\n')

    return Response(content="".join(out), media_type="text/plain; version=0.0.4; charset=utf-8")
