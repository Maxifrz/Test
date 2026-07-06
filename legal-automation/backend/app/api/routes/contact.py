"""Interne Verwaltung der Website-Kontaktanfragen (Sekretariat/Empfang)."""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from app.core.deps import DB, require_permission
from app.models.contact import ContactRequest
from app.schemas.contact import ContactRequestResponse, ContactRequestUpdate

router = APIRouter(prefix="/contact-requests", tags=["contact"])


@router.get("")
async def list_contact_requests(
    db: DB,
    status_filter: str | None = Query(None, alias="status", pattern="^(neu|erledigt)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user=Depends(require_permission("email.read")),
):
    query = select(ContactRequest)
    if status_filter:
        query = query.where(ContactRequest.status == status_filter)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    query = query.order_by(ContactRequest.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    items = [ContactRequestResponse.model_validate(r) for r in result.scalars().all()]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.patch("/{request_id}", response_model=ContactRequestResponse)
async def update_contact_request(
    request_id: int,
    data: ContactRequestUpdate,
    db: DB,
    current_user=Depends(require_permission("email.read")),
):
    result = await db.execute(select(ContactRequest).where(ContactRequest.id == request_id))
    req = result.scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anfrage nicht gefunden")
    req.status = data.status
    req.processed_by_id = current_user.id if data.status == "erledigt" else None
    await db.commit()
    await db.refresh(req)
    return req
