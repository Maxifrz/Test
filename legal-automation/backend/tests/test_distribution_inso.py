"""
Tests der InsO-Rangfolge im Verteilungsrechner.

Abgedeckt: §§ 53-55 (Masseverbindlichkeiten vorweg), § 52 (Ausfallprinzip),
§ 39 Abs. 1 Nr. 1-5 (Unterraenge), § 189 (Rueckstellung fuer Bestrittene),
§ 208 (Masseunzulaenglichkeit) und die cent-genaue Rundung.
"""
from decimal import Decimal

from app.services.distribution import (
    RANK_38,
    RANK_39,
    RANK_39_1,
    RANK_39_5,
    ClaimInput,
    compute_distribution,
)


def D(v):
    return Decimal(str(v))


# --- §§ 53-55 InsO: Masseverbindlichkeiten vorweg ---

def test_mass_liabilities_are_deducted_first():
    claims = [ClaimInput(1, D(10000)), ClaimInput(2, D(10000))]
    r = compute_distribution(D(12000), claims, mass_liabilities=D(2000))
    assert r.gross_estate == D("12000.00")
    assert r.mass_liabilities == D("2000.00")
    assert r.distributable == D("10000.00")
    # 10.000 auf 20.000 Forderungen = 50 %
    assert r.quote_38_pct == D("50.00")
    assert r.distributed_sum == D("10000.00")


def test_mass_insufficiency_yields_nothing_and_says_so():
    # § 208 InsO: Masseunzulaenglichkeit
    r = compute_distribution(D(1000), [ClaimInput(1, D(5000))], mass_liabilities=D(3000))
    assert r.distributable == D("0.00")
    assert r.distributed_sum == D("0.00")
    assert any("Masseunzul" in n for n in r.notes)


def test_without_mass_liabilities_behaviour_is_unchanged():
    claims = [ClaimInput(1, D(1000)), ClaimInput(2, D(3000))]
    r = compute_distribution(D(2000), claims)
    assert r.distributable == D("2000.00")
    assert r.quote_38_pct == D("50.00")


# --- § 52 InsO: Ausfallprinzip ---

def test_secured_creditor_participates_only_with_shortfall():
    # 10.000 Forderung, 6.000 aus dem Sicherungsgut erloest -> 4.000 Ausfall
    secured = ClaimInput(1, D(10000), secured_recovery=D(6000))
    plain = ClaimInput(2, D(6000))
    r = compute_distribution(D(5000), [secured, plain])

    item_secured = next(i for i in r.items if i.claim_id == 1)
    assert item_secured.participating_amount == D(4000)
    # 5.000 auf 10.000 teilnehmende = 50 %
    assert item_secured.amount == D("2000.00")
    assert next(i for i in r.items if i.claim_id == 2).amount == D("3000.00")
    assert any("52 InsO" in n for n in r.notes)


def test_fully_secured_creditor_gets_nothing():
    secured = ClaimInput(1, D(5000), secured_recovery=D(5000))
    plain = ClaimInput(2, D(5000))
    r = compute_distribution(D(2500), [secured, plain])
    assert next(i for i in r.items if i.claim_id == 1).amount == D("0.00")
    assert next(i for i in r.items if i.claim_id == 2).amount == D("2500.00")


def test_over_recovery_does_not_go_negative():
    # Verwertungserloes uebersteigt die Forderung -> Teilnahme 0, nicht negativ
    c = ClaimInput(1, D(1000), secured_recovery=D(1500))
    assert c.participating_amount == D(0)


# --- § 39 Abs. 1 InsO: Unterraenge ---

def test_subordinated_ranks_are_served_in_order():
    claims = [
        ClaimInput(1, D(1000), rank=RANK_38),
        ClaimInput(2, D(1000), rank=RANK_39_1),   # hoher Nachrang
        ClaimInput(3, D(1000), rank=RANK_39_5),   # Gesellschafterdarlehen
    ]
    # 1.500 verfuegbar: 1.000 an § 38, 500 bleiben fuer § 39
    r = compute_distribution(D(1500), claims)
    by_id = {i.claim_id: i for i in r.items}
    assert by_id[1].amount == D("1000.00")
    # Nr. 1 wird VOR Nr. 5 bedient -- vorher haetten sich beide 250/250 geteilt
    assert by_id[2].amount == D("500.00")
    assert by_id[3].amount == D("0.00")


def test_subordinated_only_after_full_satisfaction_of_38():
    claims = [
        ClaimInput(1, D(1000), rank=RANK_38),
        ClaimInput(2, D(1000), rank=RANK_39_1),
    ]
    r = compute_distribution(D(500), claims)
    by_id = {i.claim_id: i for i in r.items}
    assert by_id[1].amount == D("500.00")
    assert by_id[2].amount == D("0.00")


def test_generic_subordinated_rank_is_served_last():
    claims = [
        ClaimInput(1, D(500), rank=RANK_39_5),
        ClaimInput(2, D(500), rank=RANK_39),  # ohne Unterrang -> ganz hinten
    ]
    r = compute_distribution(D(500), claims)
    by_id = {i.claim_id: i for i in r.items}
    assert by_id[1].amount == D("500.00")
    assert by_id[2].amount == D("0.00")


def test_rank_quotes_are_reported():
    claims = [ClaimInput(1, D(1000), rank=RANK_38), ClaimInput(2, D(1000), rank=RANK_39_1)]
    r = compute_distribution(D(1500), claims)
    assert r.rank_quotes[RANK_38] == D("100.00")
    assert r.rank_quotes[RANK_39_1] == D("50.00")


# --- § 189 InsO: Rueckstellung fuer bestrittene Forderungen ---

def test_disputed_claim_counts_for_quota_but_is_withheld():
    claims = [
        ClaimInput(1, D(1000)),
        ClaimInput(2, D(1000), disputed=True),
    ]
    r = compute_distribution(D(1000), claims)
    by_id = {i.claim_id: i for i in r.items}
    # Quote 50 % fuer beide -- der bestrittene Anteil wird zurueckbehalten
    assert by_id[1].amount == D("500.00")
    assert by_id[2].amount == D("0.00")
    assert by_id[2].withheld is True
    assert r.withheld_sum == D("500.00")
    assert r.distributed_sum == D("500.00")
    assert any("189 InsO" in n for n in r.notes)


def test_disputed_claims_do_not_inflate_other_quotas():
    # Ohne Mitzaehlen haette Glaeubiger 1 volle 100 % bekommen und muesste
    # zurueckzahlen, sobald der Widerspruch faellt.
    claims = [ClaimInput(1, D(1000)), ClaimInput(2, D(1000), disputed=True)]
    r = compute_distribution(D(1000), claims)
    assert next(i for i in r.items if i.claim_id == 1).quote_pct == D("50.00")


# --- Rundung ---

def test_cent_exact_distribution_sums_up():
    # Drittelung: 100,00 auf drei gleiche Forderungen laesst sich nicht glatt teilen
    claims = [ClaimInput(i, D(1000)) for i in (1, 2, 3)]
    r = compute_distribution(D("100.00"), claims)
    assert sum(i.amount for i in r.items) == D("100.00")
    assert r.remainder == D("0.00")


def test_rounding_never_exceeds_the_claim():
    claims = [ClaimInput(1, D("0.01")), ClaimInput(2, D("1000"))]
    r = compute_distribution(D("1000.01"), claims)
    for item in r.items:
        assert item.amount <= item.participating_amount


def test_largest_remainder_is_deterministic():
    claims = [ClaimInput(i, D(1000)) for i in (3, 1, 2)]
    first = compute_distribution(D("100.00"), claims)
    second = compute_distribution(D("100.00"), list(reversed(claims)))
    assert {i.claim_id: i.amount for i in first.items} == {
        i.claim_id: i.amount for i in second.items
    }


def test_empty_claims_produce_empty_result():
    r = compute_distribution(D(1000), [])
    assert r.items == []
    assert r.remainder == D("1000.00")
    assert r.quote_38_pct == D("0.00")


def test_full_satisfaction_leaves_remainder():
    r = compute_distribution(D(5000), [ClaimInput(1, D(1000))])
    assert next(iter(r.items)).amount == D("1000.00")
    assert r.remainder == D("4000.00")
    assert r.quote_38_pct == D("100.00")


# --- Zusammenspiel ---

def test_combined_scenario():
    """Realistisches Verfahren: Masseverbindlichkeiten, Absonderung,
    bestrittene Forderung und Nachrang in einem Lauf."""
    claims = [
        ClaimInput(1, D(50000), rank=RANK_38),
        ClaimInput(2, D(30000), rank=RANK_38, secured_recovery=D(20000)),  # 10k Ausfall
        ClaimInput(3, D(20000), rank=RANK_38, disputed=True),
        ClaimInput(4, D(10000), rank=RANK_39_1),
    ]
    r = compute_distribution(D(100000), claims, mass_liabilities=D(20000))

    assert r.distributable == D("80000.00")
    # Teilnehmende § 38-Summe: 50.000 + 10.000 + 20.000 = 80.000 -> 100 %
    assert r.total_38 == D("80000.00")
    assert r.quote_38_pct == D("100.00")
    by_id = {i.claim_id: i for i in r.items}
    assert by_id[2].participating_amount == D(10000)
    assert by_id[3].withheld is True
    assert r.withheld_sum == D("20000.00")
    # § 38 ist voll bedient, es bleibt nichts fuer den Nachrang
    assert by_id[4].amount == D("0.00")
    assert r.distributed_sum + r.withheld_sum + r.remainder == r.distributable
