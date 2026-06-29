from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.deps import DB, get_current_user, require_permission
from app.core.rbac import Role
from app.schemas.matter import (
    MatterAccessGrant,
    MatterAccessResponse,
    MatterAccessRevoke,
    MatterCreate,
    MatterListResponse,
    MatterResponse,
    MatterUpdate,
)
from app.services import client_service, matter_service

router = APIRouter(prefix="/matters", tags=["matters"])


async def _get_accessible_matter(matter_id: int, db, current_user):
    """Return matter if user is admin or has active matter_access."""
    matter = await matter_service.get_matter(db, matter_id)
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    if current_user.role == Role.ADMIN:
        return matter

    access = await matter_service.get_matter_access(db, current_user.id, matter_id)
    if not access:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No access to this matter")

    return matter


@router.post("", response_model=MatterResponse, status_code=status.HTTP_201_CREATED)
async def create_matter(
    data: MatterCreate,
    db: DB,
    current_user=Depends(require_permission("matter.create")),
):
    client = await client_service.get_client(db, data.client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")

    matter = await matter_service.create_matter(
        db, data, created_by_id=current_user.id, client_last_name=client.last_name
    )
    return matter


@router.get("", response_model=MatterListResponse)
async def list_matters(
    db: DB,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    matter_type: str | None = Query(None),
    client_id: int | None = Query(None),
    current_user=Depends(require_permission("matter.read")),
):
    is_admin = current_user.role == Role.ADMIN
    items, total = await matter_service.list_matters_for_user(
        db,
        user_id=current_user.id,
        is_admin=is_admin,
        page=page,
        page_size=page_size,
        status=status,
        matter_type=matter_type,
        client_id=client_id,
    )
    return MatterListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{matter_id}", response_model=MatterResponse)
async def get_matter(
    matter_id: int,
    db: DB,
    current_user=Depends(require_permission("matter.read")),
):
    matter = await _get_accessible_matter(matter_id, db, current_user)
    return matter


@router.patch("/{matter_id}", response_model=MatterResponse)
async def update_matter(
    matter_id: int,
    data: MatterUpdate,
    db: DB,
    current_user=Depends(require_permission("matter.update")),
):
    matter = await _get_accessible_matter(matter_id, db, current_user)
    return await matter_service.update_matter(db, matter, data)


@router.delete("/{matter_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_matter(
    matter_id: int,
    db: DB,
    current_user=Depends(require_permission("matter.delete")),
):
    matter = await matter_service.get_matter(db, matter_id)
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")
    await matter_service.soft_delete_matter(db, matter, deleted_by_id=current_user.id)


# --- Access grant / revoke ---

@router.get("/{matter_id}/access", response_model=list[MatterAccessResponse])
async def list_access(
    matter_id: int,
    db: DB,
    current_user=Depends(require_permission("matter.read")),
):
    await _get_accessible_matter(matter_id, db, current_user)
    return await matter_service.list_matter_access(db, matter_id)


@router.post("/{matter_id}/access", response_model=MatterAccessResponse, status_code=status.HTTP_201_CREATED)
async def grant_access(
    matter_id: int,
    data: MatterAccessGrant,
    db: DB,
    current_user=Depends(require_permission("matter.access.grant")),
):
    await _get_accessible_matter(matter_id, db, current_user)
    access = await matter_service.grant_access(
        db,
        matter_id=matter_id,
        user_id=data.user_id,
        matter_role=data.matter_role,
        granted_by_id=current_user.id,
    )
    return access


@router.delete("/{matter_id}/access", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_access(
    matter_id: int,
    data: MatterAccessRevoke,
    db: DB,
    current_user=Depends(require_permission("matter.access.revoke")),
):
    await _get_accessible_matter(matter_id, db, current_user)
    revoked = await matter_service.revoke_access(db, matter_id=matter_id, user_id=data.user_id)
    if not revoked:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Access grant not found")
