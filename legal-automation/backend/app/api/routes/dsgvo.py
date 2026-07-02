import os
from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from app.core.deps import DB, require_permission
from app.models.dsgvo import (
    DataExport,
    DataRetentionPolicy,
    ErasureRequest,
    ProcessingRecord,
)
from app.models.matter import Matter
from app.models.user import User, UserSession
from app.schemas.dsgvo import (
    AdminOverviewResponse,
    DataExportResponse,
    ErasureEligibilityResponse,
    ErasureRequestCreate,
    ErasureRequestResponse,
    ProcessingRecordCreate,
    ProcessingRecordResponse,
    RetentionPolicyCreate,
    RetentionPolicyResponse,
)
from app.services import dsgvo_service
from app.services.dsgvo_retention import retention_until

router = APIRouter(prefix="/dsgvo", tags=["dsgvo"])


# --- Verarbeitungsverzeichnis (Art. 30) ---

@router.get("/vvt", response_model=list[ProcessingRecordResponse])
async def get_vvt(db: DB, current_user=Depends(require_permission("dsgvo.read"))):
    await dsgvo_service.seed_vvt_if_empty(db)
    result = await db.execute(select(ProcessingRecord).where(ProcessingRecord.is_active == True))  # noqa: E712
    return result.scalars().all()


@router.post("/vvt", response_model=ProcessingRecordResponse, status_code=status.HTTP_201_CREATED)
async def create_vvt(data: ProcessingRecordCreate, db: DB, current_user=Depends(require_permission("dsgvo.vvt.edit"))):
    rec = ProcessingRecord(**data.model_dump())
    db.add(rec)
    await db.commit()
    await db.refresh(rec)
    return rec


# --- Retention-Policies ---

@router.get("/retention-policies", response_model=list[RetentionPolicyResponse])
async def list_retention(db: DB, current_user=Depends(require_permission("dsgvo.read"))):
    result = await db.execute(select(DataRetentionPolicy))
    return result.scalars().all()


@router.post("/retention-policies", response_model=RetentionPolicyResponse, status_code=status.HTTP_201_CREATED)
async def create_retention(data: RetentionPolicyCreate, db: DB, current_user=Depends(require_permission("dsgvo.vvt.edit"))):
    pol = DataRetentionPolicy(**data.model_dump())
    db.add(pol)
    await db.commit()
    await db.refresh(pol)
    return pol


# --- Recht auf Löschung (Art. 17) ---

@router.get("/erasure-eligibility/{client_id}", response_model=ErasureEligibilityResponse)
async def erasure_eligibility(client_id: int, db: DB, current_user=Depends(require_permission("dsgvo.erasure"))):
    e = await dsgvo_service.evaluate_erasure(db, client_id)
    return ErasureEligibilityResponse(allowed=e.allowed, blocking_reasons=e.blocking_reasons)


@router.get("/erasure-requests", response_model=list[ErasureRequestResponse])
async def list_erasure(db: DB, current_user=Depends(require_permission("dsgvo.erasure"))):
    result = await db.execute(select(ErasureRequest).order_by(ErasureRequest.id.desc()))
    return result.scalars().all()


@router.post("/erasure-requests", response_model=ErasureRequestResponse, status_code=status.HTTP_201_CREATED)
async def create_erasure(data: ErasureRequestCreate, db: DB, current_user=Depends(require_permission("dsgvo.erasure"))):
    return await dsgvo_service.create_erasure_request(
        db, client_id=data.client_id, requested_by_id=current_user.id, reason=data.reason
    )


@router.post("/erasure-requests/{req_id}/execute", response_model=ErasureRequestResponse)
async def execute_erasure(req_id: int, db: DB, current_user=Depends(require_permission("dsgvo.erasure"))):
    req = await db.get(ErasureRequest, req_id)
    if not req:
        raise HTTPException(status_code=404, detail="Antrag nicht gefunden")
    if req.status == "executed":
        raise HTTPException(status_code=409, detail="Bereits ausgeführt")
    return await dsgvo_service.execute_erasure(db, req, executed_by_id=current_user.id)


@router.post("/erasure-requests/{req_id}/reject", response_model=ErasureRequestResponse)
async def reject_erasure(req_id: int, db: DB, current_user=Depends(require_permission("dsgvo.erasure"))):
    req = await db.get(ErasureRequest, req_id)
    if not req:
        raise HTTPException(status_code=404, detail="Antrag nicht gefunden")
    return await dsgvo_service.reject_erasure(db, req, decided_by_id=current_user.id)


# --- Datenportabilität (Art. 20) ---

@router.post("/export/{client_id}", response_model=DataExportResponse, status_code=status.HTTP_201_CREATED)
async def create_export(client_id: int, db: DB, current_user=Depends(require_permission("dsgvo.export"))):
    export = await dsgvo_service.create_export(db, client_id=client_id, requested_by_id=current_user.id)
    await dsgvo_service.build_export_zip(db, export)
    return DataExportResponse(
        id=export.id, client_id=export.client_id, status=export.status,
        token=export.token, expires_at=export.expires_at,
        download_path=f"/api/dsgvo/export/download/{export.token}",
    )


@router.get("/export/download/{token}")
async def download_export(token: str, db: DB, current_user=Depends(require_permission("dsgvo.export"))):
    result = await db.execute(select(DataExport).where(DataExport.token == token))
    export = result.scalar_one_or_none()
    if not export or not export.file_path:
        raise HTTPException(status_code=404, detail="Export nicht gefunden")
    if export.downloaded_at is not None:
        raise HTTPException(status_code=410, detail="Download-Link bereits verwendet (single-use)")
    if export.expires_at and datetime.now(UTC) > export.expires_at:
        export.status = "expired"
        await db.commit()
        raise HTTPException(status_code=410, detail="Download-Link abgelaufen (48 h)")
    if not os.path.exists(export.file_path):
        raise HTTPException(status_code=404, detail="Exportdatei nicht mehr vorhanden")

    export.downloaded_at = datetime.now(UTC)
    export.status = "downloaded"
    await db.commit()
    return FileResponse(export.file_path, media_type="application/zip", filename=f"datenexport_{export.client_id}.zip")


# --- Admin-Dashboard ---

@router.get("/admin/overview", response_model=AdminOverviewResponse)
async def admin_overview(db: DB, current_user=Depends(require_permission("audit.read"))):
    now = datetime.now(UTC)
    active_sessions = (await db.execute(
        select(func.count()).select_from(UserSession).where(
            UserSession.is_revoked == False, UserSession.expires_at > now  # noqa: E712
        )
    )).scalar_one()
    # Login-Lockout läuft über Redis (auth.py) — Kennzahl daher aus Redis,
    # nicht aus den (dort nicht gepflegten) users.locked_until-Spalten.
    locked_users = 0
    try:
        import redis.asyncio as aioredis
        from app.core.config import get_settings

        r = aioredis.from_url(get_settings().REDIS_URL, decode_responses=True)
        async for _ in r.scan_iter(match="login_lock:*", count=100):
            locked_users += 1
        await r.aclose()
    except Exception:
        locked_users = 0  # Redis nicht erreichbar → Kennzahl neutral
    users_total = (await db.execute(select(func.count()).select_from(User).where(User.deleted_at.is_(None)))).scalar_one()
    users_2fa = (await db.execute(
        select(func.count()).select_from(User).where(User.totp_enabled == True, User.deleted_at.is_(None))  # noqa: E712
    )).scalar_one()
    open_erasure = (await db.execute(
        select(func.count()).select_from(ErasureRequest).where(ErasureRequest.status == "open")
    )).scalar_one()
    blocked_erasure = (await db.execute(
        select(func.count()).select_from(ErasureRequest).where(ErasureRequest.status == "blocked")
    )).scalar_one()

    # Matters past retention (Kandidaten) — report only, keine Auto-Löschung
    today = date.today()
    closed = (await db.execute(
        select(Matter).where(Matter.status.in_(["closed", "archived"]), Matter.deleted_at.is_(None))
    )).scalars().all()
    past = sum(
        1 for m in closed
        if (u := retention_until(m.closed_at, m.retention_years)) is not None and today >= u
    )

    return AdminOverviewResponse(
        active_sessions=active_sessions, locked_users=locked_users, users_total=users_total,
        users_with_2fa=users_2fa, open_erasure_requests=open_erasure,
        blocked_erasure_requests=blocked_erasure, matters_past_retention=past,
    )
