"""Forderungstabelle + Verteilung: CRUD, Summen, Verteilungslauf (persistiert)."""
from __future__ import annotations

import secrets
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.insolvency import (
    STATUS_BESTRITTEN,
    STATUS_FESTGESTELLT,
    Distribution,
    DistributionItem,
    InsolvencyClaim,
)
from app.models.matter import Matter
from app.services.distribution import ClaimInput, compute_distribution


async def _next_claim_number(db: AsyncSession, matter_id: int) -> int:
    result = await db.execute(
        select(func.coalesce(func.max(InsolvencyClaim.claim_number), 0)).where(
            InsolvencyClaim.matter_id == matter_id
        )
    )
    return int(result.scalar_one()) + 1


async def create_claim(
    db: AsyncSession,
    *,
    matter_id: int,
    creditor_name: str,
    claim_amount: Decimal,
    rank: str,
    source: str = "intern",
    creditor_email: str | None = None,
    creditor_address: str | None = None,
    creditor_reference: str | None = None,
    claim_reason: str | None = None,
    created_by_id: int | None = None,
) -> InsolvencyClaim:
    claim = InsolvencyClaim(
        matter_id=matter_id,
        claim_number=await _next_claim_number(db, matter_id),
        creditor_name=creditor_name,
        creditor_email=creditor_email,
        creditor_address=creditor_address,
        creditor_reference=creditor_reference,
        claim_amount=claim_amount,
        claim_reason=claim_reason,
        rank=rank,
        source=source,
        filed_at=datetime.now(UTC),
        created_by_id=created_by_id,
    )
    db.add(claim)
    await db.commit()
    await db.refresh(claim)
    return claim


async def get_claim(db: AsyncSession, claim_id: int) -> InsolvencyClaim | None:
    result = await db.execute(
        select(InsolvencyClaim).where(
            InsolvencyClaim.id == claim_id, InsolvencyClaim.deleted_at.is_(None)
        )
    )
    return result.scalar_one_or_none()


async def list_claims(db: AsyncSession, matter_id: int) -> list[InsolvencyClaim]:
    result = await db.execute(
        select(InsolvencyClaim)
        .where(InsolvencyClaim.matter_id == matter_id, InsolvencyClaim.deleted_at.is_(None))
        .order_by(InsolvencyClaim.claim_number.asc().nullslast(), InsolvencyClaim.id.asc())
    )
    return result.scalars().all()


async def update_claim(db: AsyncSession, claim: InsolvencyClaim, updates: dict) -> InsolvencyClaim:
    for field_name, value in updates.items():
        setattr(claim, field_name, value)
    claim.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(claim)
    return claim


async def table_totals(db: AsyncSession, matter_id: int) -> dict:
    claims = await list_claims(db, matter_id)
    angemeldet = sum((Decimal(c.claim_amount) for c in claims), Decimal("0"))
    festgestellt = sum(
        (Decimal(c.established_amount or 0) for c in claims if c.status == STATUS_FESTGESTELLT),
        Decimal("0"),
    )
    return {
        "count": len(claims),
        "sum_angemeldet": angemeldet,
        "sum_festgestellt": festgestellt,
        "count_festgestellt": sum(1 for c in claims if c.status == STATUS_FESTGESTELLT),
        "count_bestritten": sum(1 for c in claims if c.status == "bestritten"),
    }


async def _established_claim_inputs(db: AsyncSession, matter_id: int) -> tuple[list[ClaimInput], dict[int, InsolvencyClaim]]:
    """
    Forderungen, die an der Verteilung teilnehmen.

    Bestrittene Forderungen sind ENTHALTEN: nach § 189 InsO zählen sie für die
    Quotenbildung mit, ihr Anteil wird aber zurückbehalten. Wer sie weglässt,
    rechnet die Quote der übrigen Gläubiger zu hoch — und muss nachverteilen,
    sobald ein Widerspruch fällt.
    """
    claims = await list_claims(db, matter_id)
    inputs: list[ClaimInput] = []
    by_id: dict[int, InsolvencyClaim] = {}
    for c in claims:
        if c.status not in (STATUS_FESTGESTELLT, STATUS_BESTRITTEN):
            continue
        # Bei bestrittenen Forderungen liegt oft kein festgestellter Betrag vor;
        # dann zählt der angemeldete Betrag als Rückstellungsgrundlage.
        amount = c.established_amount if c.established_amount is not None else c.claim_amount
        if amount is None:
            continue
        inputs.append(
            ClaimInput(
                claim_id=c.id,
                established_amount=Decimal(amount),
                rank=c.rank,
                secured_recovery=Decimal(c.secured_recovery or 0),
                disputed=c.status == STATUS_BESTRITTEN,
            )
        )
        by_id[c.id] = c
    return inputs, by_id


async def preview_distribution(
    db: AsyncSession,
    matter_id: int,
    distributable: Decimal,
    mass_liabilities: Decimal = Decimal("0"),
):
    """Berechnet eine Verteilung, ohne sie zu speichern."""
    inputs, _ = await _established_claim_inputs(db, matter_id)
    return compute_distribution(distributable, inputs, mass_liabilities=mass_liabilities)


async def run_distribution(
    db: AsyncSession, *, matter_id: int, distributable: Decimal,
    distribution_type: str, created_by_id: int,
    mass_liabilities: Decimal = Decimal("0"),
) -> Distribution:
    """Berechnet UND persistiert eine Verteilung inkl. Verteilungsverzeichnis."""
    inputs, _ = await _established_claim_inputs(db, matter_id)
    result = compute_distribution(distributable, inputs, mass_liabilities=mass_liabilities)

    dist = Distribution(
        matter_id=matter_id,
        distribution_type=distribution_type,
        distributable_amount=result.distributable,
        mass_liabilities=result.mass_liabilities,
        quote_38_pct=result.quote_38_pct,
        distributed_sum=result.distributed_sum,
        withheld_sum=result.withheld_sum,
        remainder=result.remainder,
        created_by_id=created_by_id,
    )
    db.add(dist)
    await db.flush()
    db.add_all([
        DistributionItem(
            distribution_id=dist.id,
            claim_id=item.claim_id,
            established_amount=item.established_amount,
            participating_amount=item.participating_amount,
            amount=item.amount,
            quote_pct=item.quote_pct,
            rank=item.rank,
            withheld=item.withheld,
        )
        for item in result.items
    ])
    await db.commit()
    await db.refresh(dist)
    return dist


async def enable_creditor_portal(db: AsyncSession, matter: Matter) -> str:
    """Erzeugt (falls nötig) ein Portal-Token für die Online-Forderungsanmeldung."""
    if not matter.creditor_portal_token:
        matter.creditor_portal_token = secrets.token_urlsafe(32)
        await db.commit()
        await db.refresh(matter)
    return matter.creditor_portal_token


async def matter_by_portal_token(db: AsyncSession, token: str) -> Matter | None:
    result = await db.execute(
        select(Matter).where(Matter.creditor_portal_token == token, Matter.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()
