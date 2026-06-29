"""
MT940 Parser (deutscher Bank-Subset) — eigenständig, kein externes Dependency,
kein Netzwerk-Call. Deckt die gängigen Felder ab:

  :25:  Kontoidentifikation (IBAN/Konto)
  :60F: Anfangssaldo   :62F: Schlusssaldo
  :61:  Umsatzzeile (Datum, Soll/Haben, Betrag, Referenzen)
  :86:  Mehrzweckfeld (Verwendungszweck, Gegenpartei) inkl. ?-Subfelder

Beträge im deutschen Format (Komma als Dezimaltrenner).
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from app.services.bank_statement import ParsedStatement, ParsedTransaction

# :61: Wertstellung(JJMMTT) [Buchung(MMTT)] C/D/RC/RD Betrag Auftragsschlüssel ...
_RE_61 = re.compile(
    r"^(?P<vdate>\d{6})(?P<edate>\d{4})?(?P<dc>RC|RD|C|D)(?P<amount>[\d.,]+)"
    r"(?P<rest>.*)$"
)
# :86: ?-Subfelder: ?20-?29 Verwendungszweck, ?32/?33 Name, ?31 IBAN/BLZ-Konto
_RE_SUBFIELD = re.compile(r"\?(\d{2})([^?]*)")


def _german_decimal(raw: str) -> Decimal:
    return Decimal(raw.replace(".", "").replace(",", "."))


def _yymmdd(raw: str) -> date | None:
    try:
        return date(2000 + int(raw[0:2]), int(raw[2:4]), int(raw[4:6]))
    except (ValueError, IndexError):
        return None


def _split_fields(text: str) -> list[tuple[str, str]]:
    """Split an MT940 message into (tag, value) pairs; values may span lines."""
    fields: list[tuple[str, str]] = []
    current_tag: str | None = None
    current_val: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        m = re.match(r"^:(\d{2}[A-Z]?):(.*)$", line)
        if m:
            if current_tag is not None:
                fields.append((current_tag, "\n".join(current_val)))
            current_tag = m.group(1)
            current_val = [m.group(2)]
        elif line.strip() == "-":
            continue
        elif current_tag is not None:
            current_val.append(line)
    if current_tag is not None:
        fields.append((current_tag, "\n".join(current_val)))
    return fields


def _parse_balance(value: str) -> Decimal | None:
    # Format: C/D + JJMMTT + WWW(currency) + Betrag,  z.B. "C230101EUR1.000,00"
    m = re.match(r"^([CD])(\d{6})([A-Z]{3})([\d.,]+)$", value.strip())
    if not m:
        return None
    sign = Decimal("1") if m.group(1) == "C" else Decimal("-1")
    return sign * _german_decimal(m.group(4))


def _parse_86(value: str) -> tuple[str, str | None, str | None]:
    """Return (purpose, counterparty_name, counterparty_iban) from an :86: field."""
    flat = value.replace("\n", "")
    if "?" not in flat:
        return value.replace("\n", " ").strip(), None, None
    purpose_parts: list[str] = []
    name_parts: list[str] = []
    iban: str | None = None
    for code, content in _RE_SUBFIELD.findall(flat):
        c = int(code)
        if 20 <= c <= 29:
            purpose_parts.append(content)
        elif c in (32, 33):
            name_parts.append(content)
        elif c == 31:
            if re.match(r"^[A-Z]{2}\d", content.strip()):
                iban = content.strip()
    purpose = " ".join(p.strip() for p in purpose_parts if p.strip())
    name = " ".join(n.strip() for n in name_parts if n.strip()) or None
    return purpose, name, iban


def parse_mt940(text: bytes | str) -> ParsedStatement:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")

    statement = ParsedStatement()
    fields = _split_fields(text)

    # Pair each :61: with the following :86:
    i = 0
    while i < len(fields):
        tag, value = fields[i]
        if tag == "25":
            statement.account_iban = value.strip().replace(" ", "") or None
        elif tag in ("60F", "60M"):
            statement.opening_balance = _parse_balance(value)
            cur = re.search(r"[A-Z]{3}", value)
            statement.currency = statement.currency or (cur.group(0) if cur else None)
        elif tag in ("62F", "62M"):
            statement.closing_balance = _parse_balance(value)
        elif tag == "61":
            m = _RE_61.match(value.strip())
            if m:
                dc = m.group("dc")
                direction = "in" if dc in ("C", "RD") else "out"
                tx = ParsedTransaction(
                    amount=_german_decimal(m.group("amount")),
                    direction=direction,
                    currency=statement.currency or "EUR",
                    value_date=_yymmdd(m.group("vdate")),
                )
                rest = m.group("rest")
                if "//" in rest:
                    tx.bank_reference = rest.split("//", 1)[1].strip() or None
                # Attach the following :86: if present
                if i + 1 < len(fields) and fields[i + 1][0] == "86":
                    purpose, name, iban = _parse_86(fields[i + 1][1])
                    tx.purpose = purpose
                    tx.counterparty_name = name
                    tx.counterparty_iban = iban
                    i += 1
                statement.transactions.append(tx)
        i += 1

    return statement
