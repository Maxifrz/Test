"""Texterkennung fuer hochgeladene Dokumente (Celery)."""
import asyncio

from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks_documents.extract_document_text")
def extract_document_text(document_id: int):
    """Extrahiert den Text eines Dokuments (PDF-Textschicht oder OCR)."""
    return asyncio.run(_async_extract(document_id))


async def _async_extract(document_id: int) -> dict:
    from sqlalchemy import select

    from app.core.deps import AsyncSessionLocal
    from app.models.document import Document
    from app.services import document_service

    async with AsyncSessionLocal() as db:
        doc = (await db.execute(
            select(Document).where(Document.id == document_id)
        )).scalar_one_or_none()
        if doc is None:
            return {"status": "not_found", "document_id": document_id}
        doc = await document_service.process_document_text(db, doc)
        return {
            "status": doc.ocr_status,
            "document_id": document_id,
            "chars": len(doc.extracted_text or ""),
            "from_ocr": doc.text_from_ocr,
        }
