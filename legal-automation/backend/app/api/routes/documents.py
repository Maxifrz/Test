"""
Dokumentenverwaltung mit Volltextsuche.

Aktentrennung: Jeder Endpunkt prueft `ensure_matter_access`; die Suche
filtert ueber `accessible_matter_ids`. Ohne Aktenzugriff ist ein Dokument
weder auffindbar noch herunterladbar.
"""
import logging
import os
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select

from app.core.config import get_settings
from app.core.deps import DB, accessible_matter_ids, ensure_matter_access, require_permission
from app.models.document import DOCUMENT_KINDS, Document
from app.schemas.document import (
    DocumentDetail,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdate,
)
from app.services import document_service

logger = logging.getLogger("app.documents")

router = APIRouter(prefix="/documents", tags=["documents"])


async def _load_accessible(db: DB, document_id: int, current_user) -> Document:
    doc = (await db.execute(
        select(Document).where(Document.id == document_id, Document.deleted_at.is_(None))
    )).scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dokument nicht gefunden")
    await ensure_matter_access(db, current_user, doc.matter_id)
    return doc


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    db: DB,
    file: UploadFile = File(...),
    matter_id: int = Form(...),
    title: str = Form(""),
    kind: str = Form("sonstiges"),
    document_date: date | None = Form(None),
    is_confidential: bool = Form(False),
    current_user=Depends(require_permission("matter.update")),
):
    """
    Laedt ein Dokument hoch, legt es verschluesselt ab und stoesst die
    Texterkennung als Hintergrundaufgabe an.
    """
    settings = get_settings()
    await ensure_matter_access(db, current_user, matter_id)

    if kind not in DOCUMENT_KINDS:
        raise HTTPException(status_code=422, detail=f"kind muss aus {sorted(DOCUMENT_KINDS)} sein")

    content_type = file.content_type
    if content_type not in document_service.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Dateityp {content_type} wird nicht unterstuetzt "
                   f"({sorted(document_service.ALLOWED_CONTENT_TYPES)})",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Leere Datei")
    if len(payload) > settings.DOCUMENT_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Datei groesser als {settings.DOCUMENT_MAX_BYTES // (1024*1024)} MB",
        )

    doc, created = await document_service.store_document(
        db,
        matter_id=matter_id,
        title=title or (file.filename or "Dokument"),
        kind=kind,
        filename=file.filename or "dokument",
        content_type=content_type,
        payload=payload,
        uploaded_by_id=current_user.id,
        document_date=document_date,
        is_confidential=is_confidential,
    )

    if created:
        # Texterkennung im Worker: ein 200-Seiten-Scan darf den Upload nicht
        # blockieren. Faellt Celery aus, bleibt ocr_status auf "pending" und
        # der Zustand ist sichtbar, statt still zu verschwinden.
        try:
            from app.workers.tasks_documents import extract_document_text

            extract_document_text.delay(doc.id)
        except Exception:  # pragma: no cover — Broker nicht erreichbar
            # Der Upload bleibt gueltig; ocr_status bleibt "pending" und laesst
            # sich ueber /reprocess nachholen. Nur nicht stillschweigend.
            logger.warning("OCR-Task fuer Dokument %s nicht eingereiht", doc.id, exc_info=True)

    return doc


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    db: DB,
    matter_id: int | None = Query(None),
    q: str | None = Query(None, max_length=200, description="Volltextsuche"),
    kind: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user=Depends(require_permission("matter.read")),
):
    """Volltextsuche ueber die extrahierten Texte, beschraenkt auf zugaengliche Akten."""
    if matter_id is not None:
        await ensure_matter_access(db, current_user, matter_id)
        allowed: set[int] | None = {matter_id}
    else:
        allowed = await accessible_matter_ids(db, current_user)

    items = await document_service.search_documents(
        db, query=q, matter_ids=allowed, kind=kind, limit=limit
    )
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in items], total=len(items)
    )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int,
    db: DB,
    current_user=Depends(require_permission("matter.read")),
):
    return await _load_accessible(db, document_id, current_user)


@router.get("/{document_id}/download")
async def download_document(
    document_id: int,
    db: DB,
    current_user=Depends(require_permission("matter.read")),
):
    """Entschluesselt das Dokument und liefert es aus."""
    doc = await _load_accessible(db, document_id, current_user)
    # Ein einzelner stat()-Aufruf; die eigentliche Datei liest
    # read_document_bytes bereits im Thread.
    if not os.path.exists(doc.storage_path):  # noqa: ASYNC240
        raise HTTPException(status_code=404, detail="Datei nicht mehr vorhanden")
    payload = await document_service.read_document_bytes(doc)
    return Response(
        content=payload,
        media_type=doc.content_type or "application/octet-stream",
        headers={
            # attachment + quoted filename: verhindert, dass ein hochgeladenes
            # HTML/SVG im Kanzlei-Origin ausgefuehrt wird
            "Content-Disposition": f'attachment; filename="{doc.original_filename}"'
        },
    )


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: int,
    data: DocumentUpdate,
    db: DB,
    current_user=Depends(require_permission("matter.update")),
):
    doc = await _load_accessible(db, document_id, current_user)
    for field, value in data.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(doc, field, value)
    await db.commit()
    await db.refresh(doc)
    return doc


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: int,
    db: DB,
    current_user=Depends(require_permission("matter.update")),
):
    """Texterkennung erneut anstossen (z. B. nach einem OCR-Fehler)."""
    doc = await _load_accessible(db, document_id, current_user)
    doc.ocr_status = "pending"
    doc.ocr_error = None
    await db.commit()
    try:
        from app.workers.tasks_documents import extract_document_text

        extract_document_text.delay(doc.id)
    except Exception:  # pragma: no cover
        logger.warning("OCR-Task fuer Dokument %s nicht eingereiht", doc.id, exc_info=True)
    await db.refresh(doc)
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: DB,
    current_user=Depends(require_permission("matter.update")),
):
    """
    Soft-Delete. Die verschluesselte Datei bleibt bis zur Art.-17-Loeschung
    liegen -- eine versehentlich geloeschte Handakte laesst sich sonst nicht
    wiederherstellen.
    """
    from datetime import UTC, datetime

    doc = await _load_accessible(db, document_id, current_user)
    doc.deleted_at = datetime.now(UTC)
    doc.deleted_by_id = current_user.id
    await db.commit()
