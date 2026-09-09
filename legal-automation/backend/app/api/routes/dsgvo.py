import os
from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import Integer, case, func, select, update

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
from app.services.dsgvo_retention import AO_RETENTION_YEARS

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
    """
    Single-use-Download. Die Entwertung laeuft als bedingtes UPDATE, nicht als
    Pruefung-dann-Schreiben: zwei parallele Anfragen haetten sonst beide die
    Pruefung passiert und die Datei zweimal ausgeliefert (TOCTOU).
    """
    now = datetime.now(UTC)
    claimed = await db.execute(
        update(DataExport)
        .where(
            DataExport.token == token,
            DataExport.downloaded_at.is_(None),
            DataExport.file_path.isnot(None),
        )
        .values(downloaded_at=now, status="downloaded")
        .returning(DataExport.id, DataExport.client_id, DataExport.file_path, DataExport.expires_at)
    )
    row = claimed.first()

    if row is None:
        # Nicht entwertet: entweder unbekannt oder bereits verbraucht.
        existing = (await db.execute(
            select(DataExport).where(DataExport.token == token)
        )).scalar_one_or_none()
        await db.rollback()
        if existing is None or not existing.file_path:
            raise HTTPException(status_code=404, detail="Export nicht gefunden")
        raise HTTPException(status_code=410, detail="Download-Link bereits verwendet (single-use)")

    if row.expires_at and now > row.expires_at:
        # Abgelaufen: Entwertung zuruecknehmen waere sinnlos, aber der Status
        # muss "expired" heissen statt "downloaded".
        await db.execute(
            update(DataExport).where(DataExport.id == row.id).values(status="expired")
        )
        await db.commit()
        raise HTTPException(status_code=410, detail="Download-Link abgelaufen (48 h)")

    if not os.path.exists(row.file_path):
        await db.commit()
        raise HTTPException(status_code=404, detail="Exportdatei nicht mehr vorhanden")

    await db.commit()
    return FileResponse(
        row.file_path, media_type="application/zip", filename=f"datenexport_{row.client_id}.zip"
    )


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
        from app.core.redis_client import get_redis

        async for _ in get_redis().scan_iter(match="login_lock:*", count=100):
            locked_users += 1
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

    # Matters past retention (Kandidaten) — report only, keine Auto-Löschung.
    # Als SQL-COUNT: vorher wurden ALLE geschlossenen Akten in den Speicher
    # geladen, nur um sie zu zählen.
    # Fristende = 31.12. des Jahres (Schlussjahr + n), löschbar ab dem 01.01.
    # danach (§ 50 Abs. 1 S. 2 BRAO, § 147 Abs. 4 AO) — bei steuerrelevanten
    # Akten mindestens 10 Jahre.
    past = (await db.execute(
        select(func.count()).select_from(Matter).where(
            Matter.status.in_(["closed", "archived"]),
            Matter.deleted_at.is_(None),
            Matter.closed_at.isnot(None),
            func.make_date(
                func.extract("year", Matter.closed_at).cast(Integer)
                + func.greatest(
                    Matter.retention_years,
                    case((Matter.tax_relevant.is_(True), AO_RETENTION_YEARS), else_=0),
                )
                + 1,
                1, 1,
            ) <= date.today(),
        )
    )).scalar_one()

    return AdminOverviewResponse(
        active_sessions=active_sessions, locked_users=locked_users, users_total=users_total,
        users_with_2fa=users_2fa, open_erasure_requests=open_erasure,
        blocked_erasure_requests=blocked_erasure, matters_past_retention=past,
    )
