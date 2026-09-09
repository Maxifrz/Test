"""
Bulk-Ingestion in den Wissensgraphen (V3.0), mit IngestionJob-Tracking.

Quellen:
- "gesetz":         gesetze-im-internet.de, ein Kürzel je Dokument (z. B. "inso")
- "rechtsprechung": rechtsprechung-im-internet.de, TOC → N Entscheidungen

Läuft als Celery-Task (tasks_ki.py). Embeddings werden erzeugt, wenn Ollama
erreichbar ist — sonst wird ohne Embeddings ingestiert (FTS/Graph funktionieren;
Embeddings lassen sich später nachziehen).
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.kri import service as kri_service
from app.ai.kri.adapters import gesetze_im_internet as gii
from app.ai.kri.adapters import rechtsprechung as rii
from app.models.legal_knowledge import IngestionJob


async def _get_embedder():
    """
    Liefert den Batch-Embedder oder None, wenn Ollama nicht erreichbar ist.
    Batch statt Einzelaufruf: bei einem Gesetz mit hunderten Abschnitten sind
    das hunderte HTTP-Verbindungen weniger.
    """
    from app.ai.llm.ollama_client import OllamaClient

    client = OllamaClient()
    if await client.is_available():
        return client.embed_many
    return None


async def run_gesetz_ingest(db: AsyncSession, job: IngestionJob, abbrevs: list[str]) -> None:
    """Lädt und ingestiert Gesetze; aktualisiert den Job fortlaufend."""
    embedder = await _get_embedder()
    job.status = "running"
    ingested_ids: list[int] = []
    await db.commit()
    try:
        for abbrev in abbrevs:
            law = gii.fetch_gesetz(abbrev)
            if not law.norms:
                continue
            result = await kri_service.ingest_document(
                db,
                source_type="gesetz",
                title=law.langue or law.jurabk or abbrev,
                text=gii.law_to_text(law),
                external_id=law.jurabk or abbrev.upper(),
                jurisdiction="DE",
                url_or_ref=f"{gii.BASE_URL}/{abbrev.lower()}/",
                batch_embedder=embedder,
            )
            if result.duplicate:
                job.num_duplicates += 1
            else:
                job.num_documents += 1
                job.num_chunks += result.num_chunks
                if result.document_id is not None:
                    ingested_ids.append(result.document_id)
            await db.commit()
        # Einmal am Jobende und nur fuer die neuen Dokumente statt eines
        # Full-Table-Laufs ueber den gesamten Korpus.
        if ingested_ids:
            await kri_service.resolve_citation_targets(db, document_ids=ingested_ids)
        job.status = "done"
    except Exception as exc:  # Job-Fehler protokollieren, nicht verschlucken
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"[:2000]
    job.updated_at = datetime.now(UTC)
    await db.commit()


async def run_rechtsprechung_ingest(db: AsyncSession, job: IngestionJob, limit: int = 50) -> None:
    """Lädt die neuesten Entscheidungen aus dem TOC (bis limit)."""
    embedder = await _get_embedder()
    job.status = "running"
    ingested_ids: list[int] = []
    await db.commit()
    try:
        links = rii.fetch_toc()[: max(1, min(limit, 500))]
        for link in links:
            try:
                case = rii.fetch_case(link)
            except Exception:  # noqa: S112
                continue  # einzelne defekte Downloads überspringen
            if not case.text:
                continue
            result = await kri_service.ingest_document(
                db,
                source_type="urteil",
                title=case.titel,
                text=rii.case_to_text(case),
                external_id=case.ecli or case.aktenzeichen,
                jurisdiction="DE",
                url_or_ref=link,
                batch_embedder=embedder,
            )
            if result.duplicate:
                job.num_duplicates += 1
            else:
                job.num_documents += 1
                job.num_chunks += result.num_chunks
                if result.document_id is not None:
                    ingested_ids.append(result.document_id)
            await db.commit()
        # Einmal am Jobende und nur fuer die neuen Dokumente statt eines
        # Full-Table-Laufs ueber den gesamten Korpus.
        if ingested_ids:
            await kri_service.resolve_citation_targets(db, document_ids=ingested_ids)
        job.status = "done"
    except Exception as exc:
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"[:2000]
    job.updated_at = datetime.now(UTC)
    await db.commit()
