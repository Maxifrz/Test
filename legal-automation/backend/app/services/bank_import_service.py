"""
Orchestrierung des Bankauszug-Imports: Format erkennen → parsen → Konto/
Verfahren zuordnen → dedup → kategorisieren → persistieren → Saldo-Abgleich →
ImportBatch-Report. Keine externen Calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.finance import ImportBatch, MassAccount, MassAssignmentRule, MassTransaction
from app.services import bank_assignment as ba
from app.services.bank_statement import ParsedStatement
from app.services.camt_parser import parse_camt053
from app.services.mt940_parser import parse_mt940


def detect_format(filename: str, content: bytes) -> str:
    head = content[:512].lstrip()
    if head[:5] == b"<?xml" or b"Document" in head[:200]:
        return "camt053"
    name = (filename or "").lower()
    if name.endswith((".xml",)):
        return "camt053"
    return "mt940"


def parse_statement(fmt: str, content: bytes) -> ParsedStatement:
    if fmt == "camt053":
        return parse_camt053(content)
    return parse_mt940(content)


@dataclass
class ImportReport:
    batch_id: int
    num_transactions: int
    num_assigned: int
    num_unassigned: int
    num_duplicates: int
    reconciled: bool
    statement_closing: Decimal | None
    computed_closing: Decimal | None


async def _load_rules(db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(MassAssignmentRule)
        .where(MassAssignmentRule.is_active == True)  # noqa: E712
        .order_by(MassAssignmentRule.priority.asc())
    )
    return [{"conditions": r.conditions or {}, "category": r.category} for r in result.scalars().all()]


async def _account_index(db: AsyncSession) -> dict[str, tuple[int, int]]:
    """Normalised IBAN → (mass_account_id, matter_id)."""
    result = await db.execute(
        select(MassAccount).where(MassAccount.deleted_at.is_(None), MassAccount.is_active == True)  # noqa: E712
    )
    idx: dict[str, tuple[int, int]] = {}
    for acc in result.scalars().all():
        idx[acc.iban.replace(" ", "").upper()] = (acc.id, acc.matter_id)
    return idx


async def import_statement(
    db: AsyncSession,
    *,
    filename: str,
    content: bytes,
    imported_by_id: int,
    fallback_account_id: int | None = None,
    storage_path: str | None = None,
) -> ImportReport:
    fmt = detect_format(filename, content)
    statement = parse_statement(fmt, content)
    rules = await _load_rules(db)
    accounts = await _account_index(db)

    # Resolve target account: by statement IBAN, else explicit fallback
    account_id = matter_id = None
    matched = accounts.get((statement.account_iban or "").replace(" ", "").upper())
    if matched:
        account_id, matter_id = matched
    elif fallback_account_id is not None:
        acc = await db.get(MassAccount, fallback_account_id)
        if acc is not None:
            account_id, matter_id = acc.id, acc.matter_id

    batch = ImportBatch(
        mass_account_id=account_id,
        filename=filename,
        format=fmt,
        imported_by_id=imported_by_id,
        statement_opening_balance=statement.opening_balance,
        statement_closing_balance=statement.closing_balance,
        storage_path=storage_path,
    )
    db.add(batch)
    await db.flush()

    assigned = unassigned = duplicates = 0
    for tx in statement.transactions:
        if account_id is None:
            unassigned += 1
            continue
        key = ba.dedup_key(account_id, tx)
        exists = await db.execute(
            select(MassTransaction.id).where(MassTransaction.dedup_key == key)
        )
        if exists.scalar_one_or_none() is not None:
            duplicates += 1
            continue
        category = ba.categorize(tx, rules)
        db.add(
            MassTransaction(
                mass_account_id=account_id,
                matter_id=matter_id,
                import_batch_id=batch.id,
                booking_date=tx.booking_date,
                value_date=tx.value_date,
                amount=tx.amount,
                direction=tx.direction,
                currency=tx.currency,
                purpose=tx.purpose,
                counterparty_name=tx.counterparty_name,
                counterparty_iban=tx.counterparty_iban,
                category=category,
                end_to_end_id=tx.end_to_end_id,
                bank_reference=tx.bank_reference,
                dedup_key=key,
            )
        )
        assigned += 1

    rec = ba.reconcile(statement.opening_balance, statement.transactions, statement.closing_balance)
    batch.num_transactions = len(statement.transactions)
    batch.num_assigned = assigned
    batch.num_unassigned = unassigned
    batch.num_duplicates = duplicates
    batch.computed_closing_balance = rec.computed_closing
    batch.reconciled = rec.reconciled

    await db.commit()
    await db.refresh(batch)

    return ImportReport(
        batch_id=batch.id,
        num_transactions=batch.num_transactions,
        num_assigned=assigned,
        num_unassigned=unassigned,
        num_duplicates=duplicates,
        reconciled=rec.reconciled,
        statement_closing=statement.closing_balance,
        computed_closing=rec.computed_closing,
    )
