"""Tests für den Verteilungsrechner (Insolvenzquote)."""
from decimal import Decimal

from app.services.distribution import (
    RANK_38,
    RANK_39,
    ClaimInput,
    compute_distribution,
)


def _c(cid, amount, rank=RANK_38):
    return ClaimInput(claim_id=cid, established_amount=Decimal(amount), rank=rank)


def test_simple_quota_50_percent():
    claims = [_c(1, "1000"), _c(2, "2000"), _c(3, "3000")]  # total 6000
    r = compute_distribution(Decimal("3000"), claims)
    assert r.total_38 == Decimal("6000.00")
    assert r.quote_38_pct == Decimal("50.00")
    by_id = {i.claim_id: i.amount for i in r.items}
    assert by_id == {1: Decimal("500.00"), 2: Decimal("1000.00"), 3: Decimal("1500.00")}
    assert r.distributed_sum == Decimal("3000.00")
    assert r.remainder == Decimal("0.00")


def test_rounding_difference_assigned_to_largest():
    # 3×1000, verteilbar 1000 → je 333,33 = 999,99; Restcent an größte (hier #1)
    claims = [_c(1, "1000"), _c(2, "1000"), _c(3, "1000")]
    r = compute_distribution(Decimal("1000"), claims)
    total = sum(i.amount for i in r.items)
    assert total == Decimal("1000.00")  # exakt, kein Cent verloren
    assert r.distributed_sum == Decimal("1000.00")


def test_full_satisfaction_38_then_surplus_to_39():
    claims = [_c(1, "1000"), _c(2, "1000"), _c(10, "500", RANK_39)]
    # verteilbar 2500: 2000 an §38 (100%), 500 Rest an §39
    r = compute_distribution(Decimal("2500"), claims)
    assert r.quote_38_pct == Decimal("100.00")
    by_id = {i.claim_id: i.amount for i in r.items}
    assert by_id[1] == Decimal("1000.00")
    assert by_id[2] == Decimal("1000.00")
    assert by_id[10] == Decimal("500.00")  # §39 erhält Überschuss
    assert r.remainder == Decimal("0.00")


def test_nachrangig_gets_nothing_when_38_not_full():
    claims = [_c(1, "1000"), _c(10, "500", RANK_39)]
    r = compute_distribution(Decimal("400"), claims)
    by_id = {i.claim_id: i.amount for i in r.items}
    assert by_id[1] == Decimal("400.00")   # §38 quotal (40%)
    assert by_id[10] == Decimal("0.00")    # §39 leer aus
    assert r.quote_38_pct == Decimal("40.00")


def test_surplus_beyond_all_claims_leaves_remainder():
    claims = [_c(1, "1000")]
    r = compute_distribution(Decimal("1500"), claims)
    assert r.items[0].amount == Decimal("1000.00")
    assert r.remainder == Decimal("500.00")  # mehr Masse als Forderungen


def test_no_claims_no_distribution():
    r = compute_distribution(Decimal("1000"), [])
    assert r.items == []
    assert r.quote_38_pct == Decimal("0.00")
    assert r.remainder == Decimal("1000.00")


def test_zero_distributable():
    claims = [_c(1, "1000"), _c(2, "2000")]
    r = compute_distribution(Decimal("0"), claims)
    assert all(i.amount == Decimal("0.00") for i in r.items)
    assert r.quote_38_pct == Decimal("0.00")
