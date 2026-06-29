"""
Gemeinsame, normalisierte Datentypen für Bankauszüge.

CAMT.053- und MT940-Parser liefern dieselbe Struktur (`ParsedStatement`),
damit Zuordnung/Persistenz formatunabhängig sind. Reine Datenklassen,
keine I/O — voll unit-testbar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass
class ParsedTransaction:
    amount: Decimal                       # immer positiv
    direction: str                        # "in" (Gutschrift) | "out" (Lastschrift)
    currency: str = "EUR"
    booking_date: date | None = None
    value_date: date | None = None
    purpose: str = ""                     # Verwendungszweck
    counterparty_name: str | None = None
    counterparty_iban: str | None = None
    end_to_end_id: str | None = None
    bank_reference: str | None = None

    @property
    def signed_amount(self) -> Decimal:
        return self.amount if self.direction == "in" else -self.amount


@dataclass
class ParsedStatement:
    account_iban: str | None = None
    currency: str | None = None
    opening_balance: Decimal | None = None
    closing_balance: Decimal | None = None
    transactions: list[ParsedTransaction] = field(default_factory=list)
