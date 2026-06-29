"""
Zuordnung & Kategorisierung von Bankbuchungen — reine Logik, unit-testbar.

- match_account: Auszug-IBAN → Massekonto (und damit Verfahren)
- categorize: Default nach Richtung + verfeinernde Regeln (Verwendungszweck/Gegenpartei)
- dedup_key: stabiler Hash → idempotenter Import
- reconcile: Schlusssaldo-Abgleich
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal

from app.services.bank_statement import ParsedTransaction

# Kategorien
CAT_ZUFLUSS = "massezufluss"
CAT_VERBINDLICHKEIT = "masseverbindlichkeit"
CAT_GERICHTSKOSTEN = "gerichtskosten"
CAT_VERGUETUNG = "verwaltverguetung"
CAT_SONSTIGES = "sonstiges"
CAT_UNASSIGNED = "unassigned"


def _norm_iban(iban: str | None) -> str | None:
    if not iban:
        return None
    return iban.replace(" ", "").upper()


def match_account(statement_iban: str | None, accounts: dict[str, int]) -> int | None:
    """
    accounts: Mapping normalisierte IBAN → mass_account_id.
    Gibt die mass_account_id des Auszugskontos zurück (oder None).
    """
    norm = _norm_iban(statement_iban)
    if norm is None:
        return None
    return accounts.get(norm)


def default_category(tx: ParsedTransaction) -> str:
    return CAT_ZUFLUSS if tx.direction == "in" else CAT_VERBINDLICHKEIT


def categorize(tx: ParsedTransaction, rules: list[dict] | None = None) -> str:
    """
    rules: Liste von {"conditions": {...}, "category": "..."} nach Priorität sortiert.
    conditions: purpose_contains[list], counterparty_contains[list], direction("in"/"out").
    Erste passende Regel gewinnt; sonst Default nach Richtung.
    """
    haystack_purpose = (tx.purpose or "").lower()
    haystack_cp = (tx.counterparty_name or "").lower()

    for rule in rules or []:
        cond = rule.get("conditions", {})
        if "direction" in cond and cond["direction"] != tx.direction:
            continue
        if "purpose_contains" in cond and not any(
            k.lower() in haystack_purpose for k in cond["purpose_contains"]
        ):
            continue
        if "counterparty_contains" in cond and not any(
            k.lower() in haystack_cp for k in cond["counterparty_contains"]
        ):
            continue
        # only-direction rule without keyword lists still counts as a match
        if any(k in cond for k in ("purpose_contains", "counterparty_contains", "direction")):
            return rule.get("category", default_category(tx))

    return default_category(tx)


def dedup_key(account_id: int, tx: ParsedTransaction) -> str:
    """Stabiler Hash zur Erkennung bereits importierter Buchungen."""
    ref = tx.end_to_end_id or tx.bank_reference or ""
    booking = tx.booking_date or tx.value_date
    parts = [
        str(account_id),
        ref,
        booking.isoformat() if booking else "",
        str(tx.signed_amount),
        (tx.purpose or "")[:120],
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class Reconciliation:
    opening: Decimal
    computed_closing: Decimal
    statement_closing: Decimal | None
    reconciled: bool
    difference: Decimal | None


def reconcile(
    opening: Decimal | None,
    transactions: list[ParsedTransaction],
    statement_closing: Decimal | None,
) -> Reconciliation:
    op = opening if opening is not None else Decimal("0")
    computed = op + sum((t.signed_amount for t in transactions), Decimal("0"))
    if statement_closing is None:
        return Reconciliation(op, computed, None, False, None)
    diff = computed - statement_closing
    return Reconciliation(op, computed, statement_closing, diff == 0, diff)
