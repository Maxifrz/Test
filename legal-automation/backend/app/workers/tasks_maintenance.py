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
    """Enforce data retention policies (Phase 7)."""
    pass
