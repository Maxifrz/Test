"""
CAMT.053 (ISO 20022) Parser — stdlib xml.etree, kein externes Dependency,
kein Netzwerk-Call. Namespace-robust (Vergleich über local-name).

Unterstützt die gängigen camt.053.001.02 / .08 Strukturen deutscher Banken.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date, datetime
from decimal import Decimal

from app.services.bank_statement import ParsedStatement, ParsedTransaction


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name: str) -> list:
    return [c for c in el if _local(c.tag) == name]


def _child(el, name: str):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _descendant(el, name: str):
    for d in el.iter():
        if d is not el and _local(d.tag) == name:
            return d
    return None


def _text(el) -> str | None:
    if el is not None and el.text and el.text.strip():
        return el.text.strip()
    return None


def _parse_date(el) -> date | None:
    """A BookgDt/ValDt element contains either <Dt> (date) or <DtTm> (datetime)."""
    if el is None:
        return None
    dt = _child(el, "Dt")
    if dt is None:
        dt = _child(el, "DtTm")
    raw = _text(dt)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None


def _amount(el) -> Decimal | None:
    raw = _text(el)
    return Decimal(raw) if raw else None


def _balance(stmt, code: str) -> Decimal | None:
    """Find a <Bal> with type code (e.g. OPBD=opening, CLBD=closing); apply sign."""
    for bal in _children(stmt, "Bal"):
        tp = _descendant(bal, "Cd")
        if _text(tp) != code:
            continue
        amt = _amount(_child(bal, "Amt"))
        if amt is None:
            return None
        ind = _text(_child(bal, "CdtDbtInd"))
        return -amt if ind == "DBIT" else amt
    return None


def parse_camt053(xml_bytes: bytes | str) -> ParsedStatement:
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8")
    root = ET.fromstring(xml_bytes)

    stmt = _descendant(root, "Stmt")
    if stmt is None:
        return ParsedStatement()

    acct = _child(stmt, "Acct")
    account_iban = _text(_descendant(acct, "IBAN")) if acct is not None else None

    statement = ParsedStatement(
        account_iban=account_iban,
        opening_balance=_balance(stmt, "OPBD"),
        closing_balance=_balance(stmt, "CLBD"),
    )

    for ntry in _children(stmt, "Ntry"):
        amt_el = _child(ntry, "Amt")
        amount = _amount(amt_el)
        if amount is None:
            continue
        currency = amt_el.get("Ccy", "EUR")
        direction = "in" if _text(_child(ntry, "CdtDbtInd")) == "CRDT" else "out"

        booking = _parse_date(_child(ntry, "BookgDt"))
        value = _parse_date(_child(ntry, "ValDt"))
        bank_ref = _text(_child(ntry, "AcctSvcrRef")) or _text(_child(ntry, "NtryRef"))

        tx_dtls = _descendant(ntry, "TxDtls")
        scope = tx_dtls if tx_dtls is not None else ntry

        # Verwendungszweck: alle <Ustrd> zusammenführen
        rmt = _descendant(scope, "RmtInf")
        purpose = ""
        if rmt is not None:
            purpose = " ".join(t for t in (_text(u) for u in _children(rmt, "Ustrd")) if t)

        end_to_end = _text(_descendant(scope, "EndToEndId"))

        # Gegenpartei: bei Gutschrift der Debitor, bei Lastschrift der Kreditor
        rel = _descendant(scope, "RltdPties")
        cp_name = cp_iban = None
        if rel is not None:
            party = _child(rel, "Dbtr") if direction == "in" else _child(rel, "Cdtr")
            acct_el = _child(rel, "DbtrAcct") if direction == "in" else _child(rel, "CdtrAcct")
            if party is not None:
                cp_name = _text(_descendant(party, "Nm"))
            if acct_el is not None:
                cp_iban = _text(_descendant(acct_el, "IBAN"))

        statement.currency = statement.currency or currency
        statement.transactions.append(
            ParsedTransaction(
                amount=amount,
                direction=direction,
                currency=currency,
                booking_date=booking,
                value_date=value,
                purpose=purpose,
                counterparty_name=cp_name,
                counterparty_iban=cp_iban,
                end_to_end_id=end_to_end,
                bank_reference=bank_ref,
            )
        )

    return statement
