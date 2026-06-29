"""Tests für den MT940-Parser (deutscher Subset)."""
from datetime import date
from decimal import Decimal

from app.services.mt940_parser import parse_mt940

MT940 = """:20:STARTUMS
:25:DE02100100109307118603
:28C:00012/001
:60F:C240315EUR1.000,00
:61:2403150315C500,00NTRFE2E-001//REF-001
:86:166?20Zahlung Schuldner?21Mueller?32Mueller GmbH?31DE89370400440532013000
:61:2403160316D149,50NTRFNONREF//REF-002
:86:177?20Gerichtskosten AG?21Muenchen?32Justizkasse Bayern
:62F:C240316EUR1.350,50
-"""


def test_mt940_account_and_balances():
    s = parse_mt940(MT940)
    assert s.account_iban == "DE02100100109307118603"
    assert s.currency == "EUR"
    assert s.opening_balance == Decimal("1000.00")
    assert s.closing_balance == Decimal("1350.50")
    assert len(s.transactions) == 2


def test_mt940_credit_line():
    s = parse_mt940(MT940)
    t = s.transactions[0]
    assert t.direction == "in"
    assert t.amount == Decimal("500.00")
    assert t.value_date == date(2024, 3, 15)
    assert t.bank_reference == "REF-001"
    assert "Zahlung Schuldner" in t.purpose
    assert t.counterparty_name == "Mueller GmbH"
    assert t.counterparty_iban == "DE89370400440532013000"


def test_mt940_debit_line():
    s = parse_mt940(MT940)
    t = s.transactions[1]
    assert t.direction == "out"
    assert t.amount == Decimal("149.50")
    assert t.signed_amount == Decimal("-149.50")
    assert t.counterparty_name == "Justizkasse Bayern"


def test_mt940_reconciles():
    s = parse_mt940(MT940)
    computed = s.opening_balance + sum(t.signed_amount for t in s.transactions)
    assert computed == s.closing_balance  # 1000 + 500 - 149.50 = 1350.50


def test_mt940_plain_86_without_subfields():
    msg = """:25:DE02100100109307118603
:60F:C240101EUR0,00
:61:2401010101C10,00NTRFNONREF
:86:Einfacher Verwendungszweck ohne Subfelder
:62F:C240101EUR10,00
-"""
    s = parse_mt940(msg)
    assert s.transactions[0].purpose == "Einfacher Verwendungszweck ohne Subfelder"
    assert s.transactions[0].counterparty_name is None
