from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.deps import DB, require_permission
from app.core.rbac import Role
from app.models.matter import Matter
from app.schemas.ticket import (
    CommentCreate,
    CommentResponse,
    FristTicketCreate,
    TicketCreate,
    TicketListResponse,
    TicketResponse,
    TicketUpdate,
    TimeEntryCreate,
    TimeEntryResponse,
)
from app.services import ticket_service
from app.services.deadline_calculator import FRIST_CALCULATORS
from sqlalchemy import select

router = APIRouter(prefix="/tickets", tags=["tickets"])


async def _matter_type_for(db, matter_id: int | None) -> str | None:
    if matter_id is None:
        return None
    result = await db.execute(select(Matter.matter_type).where(Matter.id == matter_id))
    return result.scalar_one_or_none()


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
async def create_ticket(
    data: TicketCreate,
    db: DB,
    current_user=Depends(require_permission("ticket.create")),
):
    matter_type = await _matter_type_for(db, data.matter_id)
    ticket = await ticket_service.create_ticket(
        db,
        title=data.title,
        description=data.description,
        created_by_id=current_user.id,
        ticket_type=data.ticket_type,
        priority=data.priority,
        due_date=data.due_date,
        matter_id=data.matter_id,
        assignee_id=data.assignee_id,
        parent_id=data.parent_id,
        recurrence_rule=data.recurrence_rule,
        matter_type=matter_type,
    )
    return ticket


@router.post("/frist", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
async def create_frist(
    data: FristTicketCreate,
    db: DB,
    current_user=Depends(require_permission("ticket.create")),
):
    """Create a deadline ticket; the due date is computed from statutory rules."""
    if data.frist_type not in FRIST_CALCULATORS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unbekannter Fristtyp. Verfügbar: {sorted(FRIST_CALCULATORS)}",
        )
    ticket = await ticket_service.create_frist_ticket(
        db,
        frist_type=data.frist_type,
        trigger_date=data.trigger_date,
        created_by_id=current_user.id,
        matter_id=data.matter_id,
        assignee_id=data.assignee_id,
    )
    return ticket


@router.get("/frist-types")
async def list_frist_types(current_user=Depends(require_permission("ticket.read"))):
    return {"frist_types": sorted(FRIST_CALCULATORS.keys())}


@router.get("", response_model=TicketListResponse)
async def list_tickets(
    db: DB,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    ticket_type: str | None = Query(None),
    matter_id: int | None = Query(None),
    assignee_id: int | None = Query(None),
    current_user=Depends(require_permission("ticket.read")),
):
    items, total = await ticket_service.list_tickets_for_user(
        db,
        user_id=current_user.id,
        is_admin=current_user.role == Role.ADMIN,
        page=page,
        page_size=page_size,
        status=status,
        ticket_type=ticket_type,
        matter_id=matter_id,
        assignee_id=assignee_id,
    )
    return TicketListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{ticket_id}", response_model=TicketResponse)
async def get_ticket(
    ticket_id: int,
    db: DB,
    current_user=Depends(require_permission("ticket.read")),
):
    ticket = await ticket_service.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return ticket


@router.patch("/{ticket_id}", response_model=TicketResponse)
async def update_ticket(
    ticket_id: int,
    data: TicketUpdate,
    db: DB,
    current_user=Depends(require_permission("ticket.update")),
):
    ticket = await ticket_service.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    updates = data.model_dump(exclude_unset=True)
    return await ticket_service.update_ticket(db, ticket, updates)


@router.delete("/{ticket_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ticket(
    ticket_id: int,
    db: DB,
    current_user=Depends(require_permission("ticket.delete")),
):
    ticket = await ticket_service.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    await ticket_service.soft_delete_ticket(db, ticket, deleted_by_id=current_user.id)


@router.post("/{ticket_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def add_comment(
    ticket_id: int,
    data: CommentCreate,
    db: DB,
    current_user=Depends(require_permission("ticket.update")),
):
    ticket = await ticket_service.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return await ticket_service.add_comment(
        db, ticket_id=ticket_id, author_id=current_user.id, body=data.body, is_internal=data.is_internal
    )


@router.post("/{ticket_id}/time", response_model=TimeEntryResponse, status_code=status.HTTP_201_CREATED)
async def add_time_entry(
    ticket_id: int,
    data: TimeEntryCreate,
    db: DB,
    current_user=Depends(require_permission("ticket.update")),
):
    ticket = await ticket_service.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return await ticket_service.add_time_entry(
        db,
        ticket_id=ticket_id,
        user_id=current_user.id,
        started_at=data.started_at,
        ended_at=data.ended_at,
        description=data.description,
        billable=data.billable,
        rate_eur=data.rate_eur,
        matter_id=data.matter_id or ticket.matter_id,
    )
