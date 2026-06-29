"""Ticket/Aufgaben service: CRUD, Frist creation, SLA computation, time tracking."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.matter_access import MatterAccess
from app.models.ticket import SLAPolicy, Ticket, TicketComment, TicketTimeEntry
from app.services import deadline_calculator as dc

URGENT_THRESHOLD_DAYS = 3


def _auto_priority(due_date: date | None, explicit: str) -> str:
    """A Frist due within 3 days becomes urgent regardless of explicit priority."""
    if due_date is not None:
        days_left = (due_date - datetime.now(UTC).date()).days
        if days_left <= URGENT_THRESHOLD_DAYS:
            return "urgent"
    return explicit


async def _resolve_sla(db: AsyncSession, matter_type: str | None, ticket_type: str) -> SLAPolicy | None:
    """Find the most specific active SLA policy for this ticket."""
    result = await db.execute(
        select(SLAPolicy).where(SLAPolicy.is_active == True)  # noqa: E712
    )
    policies = result.scalars().all()
    best: SLAPolicy | None = None
    best_score = -1
    for p in policies:
        if p.matter_type and p.matter_type != matter_type:
            continue
        if p.ticket_type and p.ticket_type != ticket_type:
            continue
        score = (1 if p.matter_type else 0) + (1 if p.ticket_type else 0)
        if score > best_score:
            best, best_score = p, score
    return best


async def create_ticket(
    db: AsyncSession,
    *,
    title: str,
    created_by_id: int,
    description: str | None = None,
    ticket_type: str = "task",
    priority: str = "normal",
    due_date: date | None = None,
    matter_id: int | None = None,
    assignee_id: int | None = None,
    parent_id: int | None = None,
    recurrence_rule: str | None = None,
    matter_type: str | None = None,
    frist_basis: str | None = None,
    frist_trigger_date: date | None = None,
    frist_calculation_note: str | None = None,
) -> Ticket:
    priority = _auto_priority(due_date, priority)

    sla = await _resolve_sla(db, matter_type, ticket_type)
    sla_due_at = None
    if sla:
        sla_due_at = datetime.now(UTC) + timedelta(hours=sla.target_hours)

    ticket = Ticket(
        title=title,
        description=description,
        ticket_type=ticket_type,
        status="open",
        priority=priority,
        due_date=due_date,
        matter_id=matter_id,
        assignee_id=assignee_id,
        created_by_id=created_by_id,
        parent_id=parent_id,
        recurrence_rule=recurrence_rule,
        sla_policy_id=sla.id if sla else None,
        sla_due_at=sla_due_at,
        frist_basis=frist_basis,
        frist_trigger_date=frist_trigger_date,
        frist_calculation_note=frist_calculation_note,
    )
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)
    return ticket


async def create_frist_ticket(
    db: AsyncSession,
    *,
    frist_type: str,
    trigger_date: date,
    created_by_id: int,
    matter_id: int | None = None,
    assignee_id: int | None = None,
    bundesland: str | None = None,
    title_prefix: str = "Frist",
) -> Ticket:
    """
    Compute a statutory deadline via the deadline_calculator and create a
    'frist' ticket with the full calculation note for audit purposes.
    """
    calc = dc.FRIST_CALCULATORS.get(frist_type)
    if calc is None:
        raise ValueError(f"Unbekannter Fristtyp: {frist_type}")

    result = calc(trigger_date, bundesland=bundesland)

    return await create_ticket(
        db,
        title=f"{title_prefix}: {result.basis}",
        description=result.note,
        created_by_id=created_by_id,
        ticket_type="frist",
        due_date=result.deadline,
        matter_id=matter_id,
        assignee_id=assignee_id,
        frist_basis=result.basis,
        frist_trigger_date=result.trigger_date,
        frist_calculation_note=result.note,
    )


async def get_ticket(db: AsyncSession, ticket_id: int) -> Ticket | None:
    result = await db.execute(
        select(Ticket).where(Ticket.id == ticket_id, Ticket.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def list_tickets_for_user(
    db: AsyncSession,
    *,
    user_id: int,
    is_admin: bool,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    ticket_type: str | None = None,
    matter_id: int | None = None,
    assignee_id: int | None = None,
) -> tuple[list[Ticket], int]:
    query = select(Ticket).where(Ticket.deleted_at.is_(None))

    # RBAC: non-admins see tickets that are unassigned-to-matter OR belong to a
    # matter they can access OR are assigned to them directly.
    if not is_admin:
        accessible = (
            select(MatterAccess.matter_id)
            .where(MatterAccess.user_id == user_id, MatterAccess.revoked_at.is_(None))
            .scalar_subquery()
        )
        query = query.where(
            (Ticket.matter_id.is_(None))
            | (Ticket.matter_id.in_(accessible))
            | (Ticket.assignee_id == user_id)
        )

    if status:
        query = query.where(Ticket.status == status)
    if ticket_type:
        query = query.where(Ticket.ticket_type == ticket_type)
    if matter_id:
        query = query.where(Ticket.matter_id == matter_id)
    if assignee_id:
        query = query.where(Ticket.assignee_id == assignee_id)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    # Order: urgent first, then by due date (nulls last)
    query = query.order_by(Ticket.due_date.asc().nullslast(), Ticket.id.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    return result.scalars().all(), total


async def update_ticket(db: AsyncSession, ticket: Ticket, updates: dict) -> Ticket:
    closing = updates.get("status") == "closed" and ticket.status != "closed"

    for field, value in updates.items():
        setattr(ticket, field, value)

    # Re-evaluate auto-urgent if due_date changed and not manually closed
    if "due_date" in updates and ticket.status not in ("closed", "blocked"):
        ticket.priority = _auto_priority(ticket.due_date, ticket.priority)

    if closing:
        ticket.closed_at = datetime.now(UTC)

    ticket.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(ticket)

    # Recurring tasks: spawn the next instance when closed
    if closing and ticket.recurrence_rule:
        await _spawn_recurrence(db, ticket)

    return ticket


async def _spawn_recurrence(db: AsyncSession, closed: Ticket) -> Ticket | None:
    """Create the next occurrence of a recurring ticket using its RRULE."""
    if not closed.due_date:
        return None
    try:
        from dateutil.rrule import rrulestr

        rule = rrulestr(closed.recurrence_rule, dtstart=datetime.combine(closed.due_date, datetime.min.time()))
        # next occurrence strictly after current due date
        after = datetime.combine(closed.due_date, datetime.min.time())
        nxt = rule.after(after)
        if nxt is None:
            return None
    except Exception:
        return None

    return await create_ticket(
        db,
        title=closed.title,
        description=closed.description,
        created_by_id=closed.created_by_id,
        ticket_type=closed.ticket_type,
        priority="normal",
        due_date=nxt.date(),
        matter_id=closed.matter_id,
        assignee_id=closed.assignee_id,
        recurrence_rule=closed.recurrence_rule,
    )


async def soft_delete_ticket(db: AsyncSession, ticket: Ticket, deleted_by_id: int) -> None:
    ticket.deleted_at = datetime.now(UTC)
    ticket.deleted_by_id = deleted_by_id
    await db.commit()


# --- Comments ---

async def add_comment(
    db: AsyncSession, *, ticket_id: int, author_id: int, body: str, is_internal: bool = True
) -> TicketComment:
    comment = TicketComment(
        ticket_id=ticket_id, author_id=author_id, body=body, is_internal=is_internal
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)
    return comment


# --- Time tracking ---

async def add_time_entry(
    db: AsyncSession,
    *,
    ticket_id: int,
    user_id: int,
    started_at: datetime,
    ended_at: datetime | None,
    description: str | None,
    billable: bool,
    rate_eur: float | None,
    matter_id: int | None,
) -> TicketTimeEntry:
    duration = None
    if ended_at:
        duration = int((ended_at - started_at).total_seconds() // 60)

    entry = TicketTimeEntry(
        ticket_id=ticket_id,
        user_id=user_id,
        matter_id=matter_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_minutes=duration,
        description=description,
        billable=billable,
        rate_eur=rate_eur,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


# --- SLA breach detection (called from Celery hourly) ---

async def find_sla_breaches(db: AsyncSession) -> list[Ticket]:
    now = datetime.now(UTC)
    result = await db.execute(
        select(Ticket).where(
            Ticket.deleted_at.is_(None),
            Ticket.sla_due_at.isnot(None),
            Ticket.sla_due_at < now,
            Ticket.sla_breached == False,  # noqa: E712
            Ticket.status.notin_(["closed"]),
        )
    )
    breached = result.scalars().all()
    for t in breached:
        t.sla_breached = True
    if breached:
        await db.commit()
    return breached
