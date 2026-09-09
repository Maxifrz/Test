from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.encryption import blind_index
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
        email_index=blind_index(data.email),
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


async def get_client_by_email(db: AsyncSession, email: str) -> Client | None:
    """Exakte Suche über den Blind-Index (verschlüsselte Spalte)."""
    idx = blind_index(email)
    if idx is None:
        return None
    result = await db.execute(
        select(Client).where(Client.email_index == idx, Client.deleted_at.is_(None))
    )
    return result.scalars().first()


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
    """
    Suche über Name, Firma und Aktenzeichen als Teilstring; über die E-Mail
    nur EXAKT (die Spalte ist verschlüsselt, gesucht wird über den
    Blind-Index). Eine Teilstring-Suche auf verschlüsselten Feldern ist
    prinzipiell nicht möglich.
    """
    query = select(Client).where(Client.deleted_at.is_(None))

    if search:
        like = f"%{search}%"
        conditions = (
            Client.last_name.ilike(like)
            | Client.first_name.ilike(like)
            | Client.company_name.ilike(like)
            | Client.client_number.ilike(like)
        )
        if "@" in search:
            conditions = conditions | (Client.email_index == blind_index(search))
        query = query.where(conditions)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    query = query.order_by(Client.last_name, Client.first_name)
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    return result.scalars().all(), total


async def update_client(db: AsyncSession, client: Client, data: ClientUpdate) -> Client:
    payload = data.model_dump(exclude_unset=True)
    for field, value in payload.items():
        setattr(client, field, value)
    # Blind-Index mitziehen, sonst zeigt die E-Mail-Zuordnung nach einer
    # Adressaenderung weiter auf den alten Wert.
    if "email" in payload:
        client.email_index = blind_index(payload["email"])
    client.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(client)
    return client


async def soft_delete_client(db: AsyncSession, client: Client, deleted_by_id: int) -> None:
    client.deleted_at = datetime.now(UTC)
    client.deleted_by_id = deleted_by_id
    await db.commit()
