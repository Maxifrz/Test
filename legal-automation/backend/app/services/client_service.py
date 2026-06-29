from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.client import Client
from app.schemas.client import ClientCreate, ClientUpdate


async def generate_client_number(db: AsyncSession) -> str:
    """Generate next client number in format M-YYYY-NNNN."""
    year = datetime.now(UTC).year
    result = await db.execute(
        select(func.count()).select_from(Client).where(
            Client.client_number.like(f"M-{year}-%")
        )
    )
    count = result.scalar_one()
    return f"M-{year}-{(count + 1):04d}"


async def create_client(db: AsyncSession, data: ClientCreate, created_by_id: int) -> Client:
    client_number = await generate_client_number(db)
    client = Client(
        client_number=client_number,
        first_name=data.first_name,
        last_name=data.last_name,
        company_name=data.company_name,
        is_company=data.is_company,
        email=data.email,
        phone=data.phone,
        address_line1=data.address_line1,
        address_line2=data.address_line2,
        postal_code=data.postal_code,
        city=data.city,
        country=data.country,
        date_of_birth=data.date_of_birth,
        tax_id=data.tax_id,
        notes=data.notes,
        dsgvo_legal_basis=data.dsgvo_legal_basis,
        dsgvo_consent_given_at=datetime.now(UTC) if data.dsgvo_legal_basis == "consent" else None,
        created_by_id=created_by_id,
    )
    db.add(client)
    await db.commit()
    await db.refresh(client)
    return client


async def get_client(db: AsyncSession, client_id: int) -> Client | None:
    result = await db.execute(
        select(Client).where(Client.id == client_id, Client.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def get_client_by_number(db: AsyncSession, client_number: str) -> Client | None:
    result = await db.execute(
        select(Client).where(Client.client_number == client_number, Client.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def list_clients(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
) -> tuple[list[Client], int]:
    query = select(Client).where(Client.deleted_at.is_(None))

    if search:
        like = f"%{search}%"
        query = query.where(
            Client.last_name.ilike(like)
            | Client.first_name.ilike(like)
            | Client.company_name.ilike(like)
            | Client.client_number.ilike(like)
            | Client.email.ilike(like)
        )

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    query = query.order_by(Client.last_name, Client.first_name)
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    return result.scalars().all(), total


async def update_client(db: AsyncSession, client: Client, data: ClientUpdate) -> Client:
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(client, field, value)
    client.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(client)
    return client


async def soft_delete_client(db: AsyncSession, client: Client, deleted_by_id: int) -> None:
    client.deleted_at = datetime.now(UTC)
    client.deleted_by_id = deleted_by_id
    await db.commit()
