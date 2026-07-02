import os
from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select, text

from app.core.config import get_settings
from app.core.deps import DB, accessible_matter_ids, ensure_matter_access, require_permission
from app.models.client import Client
from app.models.matter import Matter
from app.models.transcription import Transcription, TranscriptEdit, TranscriptSegment
from app.schemas.transcription import (
    MEETING_TYPES,
    SegmentEdit,
    SpeakerRename,
    TranscriptionDetail,
    TranscriptionListResponse,
    TranscriptionSearchHit,
)
from app.services.document_naming import build_dirname

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])

ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".aac", ".webm", ".opus"}


@router.post("", response_model=TranscriptionDetail, status_code=status.HTTP_201_CREATED)
async def upload_transcription(
    db: DB,
    title: str = Form(...),
    meeting_type: str = Form("Besprechung"),
    meeting_date: date = Form(...),
    matter_id: int | None = Form(None),
    file: UploadFile = File(...),
    current_user=Depends(require_permission("transcription.upload")),
):
    if meeting_type not in MEETING_TYPES:
        raise HTTPException(status_code=422, detail=f"meeting_type must be one of {sorted(MEETING_TYPES)}")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(status_code=422, detail=f"Unsupported audio format: {ext}")

    settings = get_settings()

    # Resolve naming context (Mandant + Aktenzeichen) for the storage directory
    mandant, aktenzeichen = "ohne-Mandant", "ohne-Akte"
    if matter_id:
        await ensure_matter_access(db, current_user, matter_id)
        result = await db.execute(select(Matter).where(Matter.id == matter_id))
        matter = result.scalar_one_or_none()
        if not matter:
            raise HTTPException(status_code=404, detail="Matter not found")
        aktenzeichen = matter.matter_number
        client_result = await db.execute(select(Client).where(Client.id == matter.client_id))
        client = client_result.scalar_one_or_none()
        if client:
            mandant = client.company_name or client.last_name or "ohne-Mandant"

    dirname = build_dirname(meeting_date, mandant, aktenzeichen, meeting_type)
    storage_dir = os.path.join(str(settings.STORAGE_ROOT), "transcriptions", dirname)
    os.makedirs(storage_dir, exist_ok=True)

    safe_name = os.path.basename(file.filename or "original")
    dest_path = os.path.join(storage_dir, safe_name)
    with open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    transcription = Transcription(
        title=title,
        meeting_type=meeting_type,
        meeting_date=meeting_date,
        matter_id=matter_id,
        created_by_id=current_user.id,
        status="queued",
        original_filename=safe_name,
        storage_dir=storage_dir,
    )
    db.add(transcription)
    await db.commit()
    await db.refresh(transcription)

    # Enqueue the Celery pipeline (import here to avoid hard celery dep at import time)
    from app.workers.tasks_transcription import process_transcription

    process_transcription.delay(transcription.id)

    return transcription


@router.get("", response_model=TranscriptionListResponse)
async def list_transcriptions(
    db: DB,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    matter_id: int | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    current_user=Depends(require_permission("transcription.read")),
):
    query = select(Transcription).where(Transcription.deleted_at.is_(None))
    if matter_id:
        await ensure_matter_access(db, current_user, matter_id)
        query = query.where(Transcription.matter_id == matter_id)
    else:
        allowed = await accessible_matter_ids(db, current_user)
        if allowed is not None:
            query = query.where(
                (Transcription.matter_id.is_(None)) | (Transcription.matter_id.in_(allowed))
            )
    if status_filter:
        query = query.where(Transcription.status == status_filter)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    query = query.order_by(Transcription.meeting_date.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    return TranscriptionListResponse(items=result.scalars().all(), total=total, page=page, page_size=page_size)


@router.get("/search", response_model=list[TranscriptionSearchHit])
async def search_transcriptions(
    db: DB,
    q: str = Query(..., min_length=2),
    current_user=Depends(require_permission("transcription.read")),
):
    """German full-text search across completed transcripts (tsvector + GIN)."""
    sql = text(
        """
        SELECT id, title, meeting_date, matter_id,
               ts_headline('german', full_text, plainto_tsquery('german', :q),
                           'MaxFragments=2, MaxWords=12, MinWords=4') AS snippet
        FROM transcriptions
        WHERE deleted_at IS NULL
          AND fts @@ plainto_tsquery('german', :q)
        ORDER BY ts_rank(fts, plainto_tsquery('german', :q)) DESC
        LIMIT 50
        """
    )
    result = await db.execute(sql, {"q": q})
    allowed = await accessible_matter_ids(db, current_user)
    return [
        TranscriptionSearchHit(
            id=row.id, title=row.title, meeting_date=row.meeting_date,
            matter_id=row.matter_id, snippet=row.snippet or "",
        )
        for row in result
        if allowed is None or row.matter_id is None or row.matter_id in allowed
    ]


@router.get("/{transcription_id}", response_model=TranscriptionDetail)
async def get_transcription(
    transcription_id: int,
    db: DB,
    current_user=Depends(require_permission("transcription.read")),
):
    result = await db.execute(
        select(Transcription).where(
            Transcription.id == transcription_id, Transcription.deleted_at.is_(None)
        )
    )
    transcription = result.scalar_one_or_none()
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    await ensure_matter_access(db, current_user, transcription.matter_id)
    return transcription


@router.patch("/{transcription_id}/segments/{segment_id}", response_model=TranscriptionDetail)
async def edit_segment(
    transcription_id: int,
    segment_id: int,
    data: SegmentEdit,
    db: DB,
    current_user=Depends(require_permission("transcription.edit")),
):
    await _ensure_transcription_access(db, current_user, transcription_id)
    result = await db.execute(
        select(TranscriptSegment).where(
            TranscriptSegment.id == segment_id,
            TranscriptSegment.transcription_id == transcription_id,
        )
    )
    segment = result.scalar_one_or_none()
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Record every change for the audit trail (who changed what, previous value)
    if data.text is not None and data.text != segment.text:
        db.add(TranscriptEdit(
            segment_id=segment.id, transcription_id=transcription_id, editor_id=current_user.id,
            field="text", old_value=segment.text, new_value=data.text,
        ))
        segment.text = data.text
        segment.edited = True
    if data.speaker_label is not None and data.speaker_label != segment.speaker_label:
        db.add(TranscriptEdit(
            segment_id=segment.id, transcription_id=transcription_id, editor_id=current_user.id,
            field="speaker_label", old_value=segment.speaker_label, new_value=data.speaker_label,
        ))
        segment.speaker_label = data.speaker_label

    await db.commit()
    return await _reload_detail(db, transcription_id)


@router.post("/{transcription_id}/rename-speaker", response_model=TranscriptionDetail)
async def rename_speaker(
    transcription_id: int,
    data: SpeakerRename,
    db: DB,
    current_user=Depends(require_permission("transcription.edit")),
):
    """Apply a human label to every segment of a given diarization speaker."""
    await _ensure_transcription_access(db, current_user, transcription_id)
    result = await db.execute(
        select(TranscriptSegment).where(
            TranscriptSegment.transcription_id == transcription_id,
            TranscriptSegment.speaker == data.speaker,
        )
    )
    segments = result.scalars().all()
    if not segments:
        raise HTTPException(status_code=404, detail="No segments for that speaker")
    for seg in segments:
        seg.speaker_label = data.label
    await db.commit()
    return await _reload_detail(db, transcription_id)


@router.delete("/{transcription_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transcription(
    transcription_id: int,
    db: DB,
    current_user=Depends(require_permission("transcription.delete")),
):
    result = await db.execute(
        select(Transcription).where(
            Transcription.id == transcription_id, Transcription.deleted_at.is_(None)
        )
    )
    transcription = result.scalar_one_or_none()
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    await ensure_matter_access(db, current_user, transcription.matter_id)
    transcription.deleted_at = datetime.now(UTC)
    transcription.deleted_by_id = current_user.id
    await db.commit()


async def _ensure_transcription_access(db, user, transcription_id: int) -> None:
    result = await db.execute(
        select(Transcription).where(
            Transcription.id == transcription_id, Transcription.deleted_at.is_(None)
        )
    )
    transcription = result.scalar_one_or_none()
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    await ensure_matter_access(db, user, transcription.matter_id)


async def _reload_detail(db, transcription_id: int) -> Transcription:
    result = await db.execute(
        select(Transcription).where(Transcription.id == transcription_id)
    )
    return result.scalar_one()
