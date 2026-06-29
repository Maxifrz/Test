"""Massekonten: CRUD + Massebestand + Transaktionsabfrage."""
from __future__ import annotations

from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.finance import MassAccount, MassTransaction


async def create_account(db: AsyncSession, *, matter_id: int, iban: str, created_by_id: int,
                         bic: str | None = None, bank_name: str | None = None,
                         account_label: str | None = None, account_type: str = "sonderkonto",
                         opening_balance: Decimal = Decimal("0")) -> MassAccount:
    account = MassAccount(
        matter_id=matter_id, iban=iban.replace(" ", "").upper(), bic=bic, bank_name=bank_name,
        account_label=account_label, account_type=account_type,
        opening_balance=opening_balance, created_by_id=created_by_id,
    )
    db.add(account)
    await db.commit()
    await db.refresh(account)
    return account


async def get_account(db: AsyncSession, account_id: int) -> MassAccount | None:
    result = await db.execute(
        select(MassAccount).where(MassAccount.id == account_id, MassAccount.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def list_accounts(db: AsyncSession, matter_id: int | None = None) -> list[MassAccount]:
    query = select(MassAccount).where(MassAccount.deleted_at.is_(None))
    if matter_id:
        query = query.where(MassAccount.matter_id == matter_id)
    result = await db.execute(query.order_by(MassAccount.id))
    return result.scalars().all()


async def current_balance(db: AsyncSession, account_id: int) -> Decimal:
    """Massebestand = Anfangssaldo + Σ(Zuflüsse) − Σ(Abflüsse)."""
    acc = await get_account(db, account_id)
    if acc is None:
        return Decimal("0")
    inflow = await db.execute(
        select(func.coalesce(func.sum(MassTransaction.amount), 0)).where(
            MassTransaction.mass_account_id == account_id, MassTransaction.direction == "in"
        )
    )
    outflow = await db.execute(
        select(func.coalesce(func.sum(MassTransaction.amount), 0)).where(
            MassTransaction.mass_account_id == account_id, MassTransaction.direction == "out"
        )
    )
    return Decimal(acc.opening_balance) + Decimal(inflow.scalar_one()) - Decimal(outflow.scalar_one())


async def list_transactions(
    db: AsyncSession, *, account_id: int | None = None, matter_id: int | None = None,
    category: str | None = None, page: int = 1, page_size: int = 50,
) -> tuple[list[MassTransaction], int]:
    query = select(MassTransaction)
    if account_id:
        query = query.where(MassTransaction.mass_account_id == account_id)
    if matter_id:
        query = query.where(MassTransaction.matter_id == matter_id)
    if category:
        query = query.where(MassTransaction.category == category)

    count = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count.scalar_one()

    query = query.order_by(MassTransaction.booking_date.desc().nullslast(), MassTransaction.id.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    return result.scalars().all(), total
