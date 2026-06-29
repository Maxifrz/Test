from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.deps import DB, get_current_user, require_permission
from app.schemas.client import ClientCreate, ClientListResponse, ClientResponse, ClientUpdate
from app.services import client_service

router = APIRouter(prefix="/clients", tags=["clients"])


@router.post("", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
async def create_client(
    data: ClientCreate,
    db: DB,
    current_user=Depends(require_permission("client.create")),
):
    client = await client_service.create_client(db, data, created_by_id=current_user.id)
    return client


@router.get("", response_model=ClientListResponse)
async def list_clients(
    db: DB,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None, max_length=100),
    current_user=Depends(require_permission("client.read")),
):
    items, total = await client_service.list_clients(db, page=page, page_size=page_size, search=search)
    return ClientListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: int,
    db: DB,
    current_user=Depends(require_permission("client.read")),
):
    client = await client_service.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    return client


@router.patch("/{client_id}", response_model=ClientResponse)
async def update_client(
    client_id: int,
    data: ClientUpdate,
    db: DB,
    current_user=Depends(require_permission("client.update")),
):
    client = await client_service.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    return await client_service.update_client(db, client, data)


@router.delete("/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_client(
    client_id: int,
    db: DB,
    current_user=Depends(require_permission("client.delete")),
):
    client = await client_service.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    await client_service.soft_delete_client(db, client, deleted_by_id=current_user.id)
