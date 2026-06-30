"""Maintenance tasks: SLA checks, retention cleanup, session cleanup."""
import asyncio

from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks_maintenance.check_sla_breaches")
def check_sla_breaches():
    """Flag tickets whose SLA target has elapsed (Phase 3)."""
    return asyncio.run(_async_check_sla())


async def _async_check_sla() -> dict:
    from app.core.deps import AsyncSessionLocal
    from app.services import ticket_service

    async with AsyncSessionLocal() as db:
        breached = await ticket_service.find_sla_breaches(db)
    return {"status": "ok", "breached_count": len(breached)}


@celery_app.task(name="app.workers.tasks_maintenance.run_retention_cleanup")
def run_retention_cleanup():
    """
    Nächtliche Retention-Prüfung (Phase 7). DSGVO-sicher: meldet nur Akten,
    deren gesetzliche Aufbewahrungsfrist abgelaufen ist (Löschkandidaten) —
    es wird NICHTS automatisch gelöscht. Die eigentliche Löschung erfolgt über
    den geprüften Erasure-Workflow mit menschlicher Freigabe.
    """
    return asyncio.run(_async_retention_report())


async def _async_retention_report() -> dict:
    from datetime import date

    from sqlalchemy import select

    from app.core.deps import AsyncSessionLocal
    from app.models.matter import Matter
    from app.services.dsgvo_retention import retention_until

    today = date.today()
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Matter).where(
                Matter.status.in_(["closed", "archived"]), Matter.deleted_at.is_(None)
            )
        )
        candidates = [
            m.matter_number
            for m in result.scalars().all()
            if (u := retention_until(m.closed_at, m.retention_years)) is not None and today >= u
        ]
    return {"status": "ok", "deletion_candidates": len(candidates), "matters": candidates}
