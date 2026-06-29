"""
Verteilungsrechner (Insolvenzquote) — reine Logik, unit-testbar.

Die verteilbare Masse wird auf die **festgestellten** Insolvenzforderungen
(§ 38 InsO) quotal verteilt. Erst wenn alle § 38-Forderungen voll befriedigt
sind, fließt ein Überschuss an nachrangige Gläubiger (§ 39 InsO).

Cent-genau: Rundungsdifferenzen werden der betragsgrößten Forderung zugewiesen,
sodass die Summe der Auszahlungen exakt der verteilbaren Masse (bzw. der
Forderungssumme bei Vollbefriedigung) entspricht.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")

RANK_38 = "insolvenz_38"      # reguläre Insolvenzgläubiger
RANK_39 = "nachrangig_39"     # nachrangige Insolvenzgläubiger
# absonderung/aussonderung/masseverbindlichkeit nehmen NICHT an der Quote teil


@dataclass
class ClaimInput:
    claim_id: int
    established_amount: Decimal   # festgestellter Betrag
    rank: str = RANK_38


@dataclass
class DistributionItem:
    claim_id: int
    established_amount: Decimal
    amount: Decimal
    quote_pct: Decimal


@dataclass
class DistributionResult:
    distributable: Decimal
    total_38: Decimal
    total_39: Decimal
    quote_38_pct: Decimal               # Prozentsatz für § 38 (0–100)
    items: list[DistributionItem] = field(default_factory=list)
    distributed_sum: Decimal = Decimal("0.00")
    remainder: Decimal = Decimal("0.00")  # nicht verteilte Restmasse


def _money(v: Decimal) -> Decimal:
    return v.quantize(CENT, rounding=ROUND_HALF_UP)


def _distribute_prorata(amount_available: Decimal, claims: list[ClaimInput]) -> tuple[dict[int, Decimal], Decimal]:
    """
    Verteilt amount_available quotal auf claims (nach established_amount).
    Gibt {claim_id: Betrag} und die effektiv verteilte Summe zurück.
    Rundungsdifferenz geht an die größte Forderung.
    """
    total = sum((c.established_amount for c in claims), Decimal("0"))
    if total <= 0 or amount_available <= 0:
        return {c.claim_id: Decimal("0.00") for c in claims}, Decimal("0.00")

    capped = min(amount_available, total)
    quote = capped / total
    amounts: dict[int, Decimal] = {}
    for c in claims:
        amounts[c.claim_id] = _money(c.established_amount * quote)

    distributed = sum(amounts.values(), Decimal("0"))
    diff = _money(capped) - distributed
    if diff != 0 and claims:
        largest = max(claims, key=lambda c: c.established_amount)
        amounts[largest.claim_id] = _money(amounts[largest.claim_id] + diff)
        distributed = _money(capped)
    return amounts, distributed


def compute_distribution(distributable: Decimal, claims: list[ClaimInput]) -> DistributionResult:
    """
    claims: NUR festgestellte Forderungen (Filterung durch den Aufrufer).
    """
    claims_38 = [c for c in claims if c.rank == RANK_38]
    claims_39 = [c for c in claims if c.rank == RANK_39]
    total_38 = sum((c.established_amount for c in claims_38), Decimal("0"))
    total_39 = sum((c.established_amount for c in claims_39), Decimal("0"))

    distributable = _money(distributable)
    items: list[DistributionItem] = []
    distributed = Decimal("0.00")

    # Stufe 1: § 38
    amounts_38, dist_38 = _distribute_prorata(distributable, claims_38)
    distributed += dist_38

    if total_38 > 0 and distributable >= total_38:
        quote_38_pct = Decimal("100.00")
    elif total_38 > 0:
        quote_38_pct = _money(distributable / total_38 * 100)
    else:
        quote_38_pct = Decimal("0.00")

    for c in claims_38:
        amt = amounts_38.get(c.claim_id, Decimal("0.00"))
        q = _money(amt / c.established_amount * 100) if c.established_amount > 0 else Decimal("0.00")
        items.append(DistributionItem(c.claim_id, c.established_amount, amt, q))

    # Stufe 2: § 39 nur bei Überschuss
    rest = distributable - dist_38
    amounts_39, dist_39 = ({}, Decimal("0.00"))
    if rest > 0 and claims_39:
        amounts_39, dist_39 = _distribute_prorata(rest, claims_39)
        distributed += dist_39
    for c in claims_39:
        amt = amounts_39.get(c.claim_id, Decimal("0.00"))
        q = _money(amt / c.established_amount * 100) if c.established_amount > 0 else Decimal("0.00")
        items.append(DistributionItem(c.claim_id, c.established_amount, amt, q))

    return DistributionResult(
        distributable=distributable,
        total_38=_money(total_38),
        total_39=_money(total_39),
        quote_38_pct=quote_38_pct,
        items=items,
        distributed_sum=_money(distributed),
        remainder=_money(distributable - distributed),
    )
