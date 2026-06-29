from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rbac import MatterRole
from app.models.matter import Matter
from app.models.matter_access import MatterAccess
from app.schemas.matter import MatterCreate, MatterUpdate


async def generate_matter_number(db: AsyncSession, client_last_name: str) -> str:
    """Generate next matter number in format YYYY-NNNN-XXX (XXX = first 3 chars of last name)."""
    year = datetime.now(UTC).year
    suffix = client_last_name[:3].upper()
    result = await db.execute(
        select(func.count()).select_from(Matter).where(
            Matter.matter_number.like(f"{year}-%")
        )
    )
    count = result.scalar_one()
    return f"{year}-{(count + 1):04d}-{suffix}"


async def create_matter(
    db: AsyncSession,
    data: MatterCreate,
    created_by_id: int,
    client_last_name: str,
) -> Matter:
    matter_number = await generate_matter_number(db, client_last_name)
    now = datetime.now(UTC)

    matter = Matter(
        matter_number=matter_number,
        title=data.title,
        matter_type=data.matter_type,
        status="open",
        client_id=data.client_id,
        lead_anwalt_id=data.lead_anwalt_id,
        created_by_id=created_by_id,
        court_file_ref=data.court_file_ref,
        court_name=data.court_name,
        opposing_party=data.opposing_party,
        opposing_counsel=data.opposing_counsel,
        statute_of_limitations=data.statute_of_limitations,
        retention_years=data.retention_years,
        description=data.description,
        opened_at=now,
    )
    db.add(matter)
    await db.flush()  # get matter.id without committing

    # Lead anwalt gets automatic LEAD access
    access = MatterAccess(
        user_id=data.lead_anwalt_id,
        matter_id=matter.id,
        matter_role=MatterRole.LEAD,
        granted_by_id=created_by_id,
        granted_at=now,
    )
    db.add(access)

    # Creator also gets LEAD access if different from lead_anwalt
    if created_by_id != data.lead_anwalt_id:
        creator_access = MatterAccess(
            user_id=created_by_id,
            matter_id=matter.id,
            matter_role=MatterRole.LEAD,
            granted_by_id=created_by_id,
            granted_at=now,
        )
        db.add(creator_access)

    await db.commit()
    await db.refresh(matter)
    return matter


async def get_matter(db: AsyncSession, matter_id: int) -> Matter | None:
    result = await db.execute(
        select(Matter).where(Matter.id == matter_id, Matter.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def get_matter_by_number(db: AsyncSession, matter_number: str) -> Matter | None:
    result = await db.execute(
        select(Matter).where(Matter.matter_number == matter_number, Matter.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def get_matter_access(db: AsyncSession, user_id: int, matter_id: int) -> MatterAccess | None:
    result = await db.execute(
        select(MatterAccess).where(
            MatterAccess.user_id == user_id,
            MatterAccess.matter_id == matter_id,
            MatterAccess.revoked_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def list_matters_for_user(
    db: AsyncSession,
    user_id: int,
    is_admin: bool,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    matter_type: str | None = None,
    client_id: int | None = None,
) -> tuple[list[Matter], int]:
    query = select(Matter).where(Matter.deleted_at.is_(None))

    if not is_admin:
        # Non-admins only see matters where they have active access
        accessible = (
            select(MatterAccess.matter_id)
            .where(MatterAccess.user_id == user_id, MatterAccess.revoked_at.is_(None))
            .scalar_subquery()
        )
        query = query.where(Matter.id.in_(accessible))

    if status:
        query = query.where(Matter.status == status)
    if matter_type:
        query = query.where(Matter.matter_type == matter_type)
    if client_id:
        query = query.where(Matter.client_id == client_id)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    query = query.order_by(Matter.opened_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    return result.scalars().all(), total


async def update_matter(db: AsyncSession, matter: Matter, data: MatterUpdate) -> Matter:
    updates = data.model_dump(exclude_unset=True)

    if "status" in updates and updates["status"] == "closed" and not matter.closed_at:
        matter.closed_at = datetime.now(UTC)

    for field, value in updates.items():
        setattr(matter, field, value)
    matter.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(matter)
    return matter


async def grant_access(
    db: AsyncSession,
    matter_id: int,
    user_id: int,
    matter_role: MatterRole,
    granted_by_id: int,
) -> MatterAccess:
    existing = await get_matter_access(db, user_id, matter_id)
    if existing:
        existing.matter_role = matter_role
        existing.revoked_at = None
        existing.granted_by_id = granted_by_id
        existing.granted_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(existing)
        return existing

    access = MatterAccess(
        user_id=user_id,
        matter_id=matter_id,
        matter_role=matter_role,
        granted_by_id=granted_by_id,
        granted_at=datetime.now(UTC),
    )
    db.add(access)
    await db.commit()
    await db.refresh(access)
    return access


async def revoke_access(
    db: AsyncSession,
    matter_id: int,
    user_id: int,
) -> bool:
    access = await get_matter_access(db, user_id, matter_id)
    if not access:
        return False
    access.revoked_at = datetime.now(UTC)
    await db.commit()
    return True


async def list_matter_access(db: AsyncSession, matter_id: int) -> list[MatterAccess]:
    result = await db.execute(
        select(MatterAccess).where(
            MatterAccess.matter_id == matter_id,
            MatterAccess.revoked_at.is_(None),
        )
    )
    return result.scalars().all()


async def soft_delete_matter(db: AsyncSession, matter: Matter, deleted_by_id: int) -> None:
    matter.deleted_at = datetime.now(UTC)
    matter.deleted_by_id = deleted_by_id
    await db.commit()
