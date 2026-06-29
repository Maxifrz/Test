"""Tests für Zuordnung, Kategorisierung, Dedup und Saldo-Abgleich."""
from datetime import date
from decimal import Decimal

from app.services.bank_assignment import (
    CAT_GERICHTSKOSTEN,
    CAT_VERBINDLICHKEIT,
    CAT_ZUFLUSS,
    categorize,
    dedup_key,
    match_account,
    reconcile,
)
from app.services.bank_statement import ParsedTransaction


def _tx(**kw):
    base = dict(amount=Decimal("100.00"), direction="in")
    base.update(kw)
    return ParsedTransaction(**base)


def test_match_account_normalizes_iban():
    accounts = {"DE02100100109307118603": 42}
    assert match_account("DE02 1001 0010 9307 1186 03", accounts) == 42
    assert match_account("de02100100109307118603", accounts) == 42
    assert match_account("DE99999999999999999999", accounts) is None
    assert match_account(None, accounts) is None


def test_default_categories_by_direction():
    assert categorize(_tx(direction="in")) == CAT_ZUFLUSS
    assert categorize(_tx(direction="out")) == CAT_VERBINDLICHKEIT


def test_rule_refines_category():
    rules = [
        {"conditions": {"direction": "out", "purpose_contains": ["Gerichtskosten"]},
         "category": CAT_GERICHTSKOSTEN},
    ]
    tx = _tx(direction="out", purpose="Gerichtskosten AG München")
    assert categorize(tx, rules) == CAT_GERICHTSKOSTEN
    # Non-matching purpose falls back to default
    assert categorize(_tx(direction="out", purpose="Miete"), rules) == CAT_VERBINDLICHKEIT


def test_rule_counterparty_match():
    rules = [{"conditions": {"counterparty_contains": ["Justizkasse"]},
              "category": CAT_GERICHTSKOSTEN}]
    assert categorize(_tx(direction="out", counterparty_name="Justizkasse Bayern"), rules) == CAT_GERICHTSKOSTEN


def test_dedup_key_stable_and_distinct():
    a = _tx(end_to_end_id="E2E-1", booking_date=date(2024, 3, 15), amount=Decimal("500.00"))
    b = _tx(end_to_end_id="E2E-1", booking_date=date(2024, 3, 15), amount=Decimal("500.00"))
    c = _tx(end_to_end_id="E2E-2", booking_date=date(2024, 3, 15), amount=Decimal("500.00"))
    assert dedup_key(1, a) == dedup_key(1, b)      # identical → same key (idempotent)
    assert dedup_key(1, a) != dedup_key(1, c)      # different ref → different key
    assert dedup_key(1, a) != dedup_key(2, a)      # different account → different key


def test_reconcile_match():
    txs = [_tx(direction="in", amount=Decimal("500.00")),
           _tx(direction="out", amount=Decimal("149.50"))]
    r = reconcile(Decimal("1000.00"), txs, Decimal("1350.50"))
    assert r.computed_closing == Decimal("1350.50")
    assert r.reconciled is True
    assert r.difference == Decimal("0")


def test_reconcile_mismatch_flagged():
    txs = [_tx(direction="in", amount=Decimal("500.00"))]
    r = reconcile(Decimal("1000.00"), txs, Decimal("1400.00"))
    assert r.reconciled is False
    assert r.difference == Decimal("100.00")
