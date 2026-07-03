"""Celery-Tasks der KI-Rechtsrecherche: Bulk-Ingestion öffentlicher Rechtsquellen."""
import asyncio

from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks_ki.run_bulk_ingest")
def run_bulk_ingest(job_id: int, source: str, params: dict):
    """
    source: "gesetz" (params: {"abbrevs": ["inso", ...]})
          | "rechtsprechung" (params: {"limit": 50})
    """
    return asyncio.run(_async_bulk_ingest(job_id, source, params))


async def _async_bulk_ingest(job_id: int, source: str, params: dict) -> dict:
    from app.ai.kri import bulk_ingest
    from app.core.deps import AsyncSessionLocal
    from app.models.legal_knowledge import IngestionJob

    async with AsyncSessionLocal() as db:
        job = await db.get(IngestionJob, job_id)
        if job is None:
            return {"status": "error", "detail": f"Job {job_id} nicht gefunden"}

        if source == "gesetz":
            await bulk_ingest.run_gesetz_ingest(db, job, params.get("abbrevs", []))
        elif source == "rechtsprechung":
            await bulk_ingest.run_rechtsprechung_ingest(db, job, int(params.get("limit", 50)))
        else:
            job.status = "failed"
            job.error = f"Unbekannte Quelle: {source}"
            await db.commit()

        return {
            "status": job.status,
            "documents": job.num_documents,
            "chunks": job.num_chunks,
            "duplicates": job.num_duplicates,
        }
