"""
KI-Rechtsrecherche (V3.0 GraphRAG): quellenbelegte Rechtsauskunft, Ingestion,
Feedback. Alle Ausgaben sind ENTWÜRFE (Human-in-the-loop, RDG) und werden in
ki_queries auditiert. Interne Dokumente unterliegen matter_access.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.deps import DB, accessible_matter_ids, ensure_matter_access, require_permission
from app.models.legal_knowledge import IngestionJob, KiQuery, LegalChunk, LegalDocument
from app.schemas.ki import (
    IngestionJobResponse,
    KiBulkIngestRequest,
    KiDocumentResponse,
    KiFeedbackRequest,
    KiIngestRequest,
    KiIngestResponse,
    KiQueryRequest,
    KiQueryResponse,
    KiSource,
    KiStatusResponse,
)
from app.ai.kri import service as kri_service

router = APIRouter(prefix="/ki", tags=["ki"])


def _require_enabled():
    if not get_settings().KI_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KI-Modul ist deaktiviert (KI_ENABLED=false)",
        )


@router.get("/status", response_model=KiStatusResponse)
async def ki_status(db: DB, current_user=Depends(require_permission("ki.use"))):
    settings = get_settings()
    available = False
    if settings.KI_ENABLED:
        from app.ai.llm.ollama_client import OllamaClient
        available = await OllamaClient().is_available()
    num_docs = (await db.execute(
        select(func.count()).select_from(LegalDocument).where(LegalDocument.is_active == True)  # noqa: E712
    )).scalar_one()
    num_chunks = (await db.execute(select(func.count()).select_from(LegalChunk))).scalar_one()
    return KiStatusResponse(
        enabled=settings.KI_ENABLED, ollama_available=available,
        llm_model=settings.KI_LLM_MODEL, embed_model=settings.KI_EMBED_MODEL,
        num_documents=num_docs, num_chunks=num_chunks,
    )


@router.post("/query", response_model=KiQueryResponse)
async def query(
    data: KiQueryRequest,
    db: DB,
    current_user=Depends(require_permission("ki.use")),
):
    _require_enabled()
    if data.matter_id:
        await ensure_matter_access(db, current_user, data.matter_id)
    allowed = await accessible_matter_ids(db, current_user)

    from app.ai.llm.ollama_client import OllamaError

    try:
        result = await kri_service.query_knowledge(
            db, question=data.question, user_id=current_user.id,
            matter_id=data.matter_id, allowed_matter_ids=allowed,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM nicht erreichbar: {exc}")

    return KiQueryResponse(
        query_id=result.query_id, answer=result.answer, grounded=result.grounded,
        sources=[KiSource(**s) for s in result.sources], model=result.model,
    )


@router.post("/ingest", response_model=KiIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    data: KiIngestRequest,
    db: DB,
    current_user=Depends(require_permission("ki.admin")),
):
    _require_enabled()
    if data.matter_id:
        await ensure_matter_access(db, current_user, data.matter_id)
    # interne Quellen MÜSSEN aktengebunden sein (RBAC-Anker)
    if data.source_type.startswith("intern_") and not data.matter_id:
        raise HTTPException(status_code=422, detail="Interne Quellen erfordern matter_id")

    from app.ai.llm.ollama_client import OllamaClient, OllamaError

    client = OllamaClient()
    embedder = client.embed if await client.is_available() else None

    try:
        result = await kri_service.ingest_document(
            db, source_type=data.source_type, title=data.title, text=data.text,
            external_id=data.external_id, jurisdiction=data.jurisdiction,
            url_or_ref=data.url_or_ref, matter_id=data.matter_id, embedder=embedder,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    await kri_service.resolve_citation_targets(db)
    return KiIngestResponse(
        document_id=result.document_id, num_chunks=result.num_chunks, duplicate=result.duplicate
    )


@router.post("/ingest-bulk", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_bulk(
    data: KiBulkIngestRequest,
    db: DB,
    current_user=Depends(require_permission("ki.admin")),
):
    """Startet einen Bulk-Ingest öffentlicher Rechtsquellen als Celery-Job.
    gesetz: abbrevs = Kürzel von gesetze-im-internet.de (z. B. ["inso"]).
    rechtsprechung: neueste Entscheidungen aus dem RII-TOC (limit)."""
    _require_enabled()
    if data.source == "gesetz" and not data.abbrevs:
        raise HTTPException(status_code=422, detail="abbrevs erforderlich für source=gesetz")

    job = IngestionJob(source=data.source, status="pending", created_by_id=current_user.id)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from app.workers.tasks_ki import run_bulk_ingest

    params = {"abbrevs": data.abbrevs} if data.source == "gesetz" else {"limit": data.limit}
    run_bulk_ingest.delay(job.id, data.source, params)
    return job


@router.get("/ingestion-jobs", response_model=list[IngestionJobResponse])
async def list_ingestion_jobs(db: DB, current_user=Depends(require_permission("ki.admin"))):
    result = await db.execute(select(IngestionJob).order_by(IngestionJob.id.desc()).limit(50))
    return result.scalars().all()


@router.get("/documents", response_model=list[KiDocumentResponse])
async def list_documents(db: DB, current_user=Depends(require_permission("ki.use"))):
    query = select(LegalDocument).where(LegalDocument.is_active == True)  # noqa: E712
    allowed = await accessible_matter_ids(db, current_user)
    if allowed is not None:
        query = query.where(
            (LegalDocument.matter_id.is_(None)) | (LegalDocument.matter_id.in_(allowed))
        )
    result = await db.execute(query.order_by(LegalDocument.id.desc()).limit(200))
    return result.scalars().all()


@router.post("/queries/{query_id}/feedback")
async def feedback(
    query_id: int,
    data: KiFeedbackRequest,
    db: DB,
    current_user=Depends(require_permission("ki.use")),
):
    q = await db.get(KiQuery, query_id)
    if not q:
        raise HTTPException(status_code=404, detail="Anfrage nicht gefunden")
    if q.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Nur eigene Anfragen bewertbar")
    q.feedback = data.feedback
    q.feedback_note = data.note
    await db.commit()
    return {"detail": "Feedback gespeichert"}
