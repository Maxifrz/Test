"""
Tests der insolvenzrechtlichen Fristen.

Der Schwerpunkt der Kanzlei ist Insolvenzrecht, die Kalkulatoren deckten aber
nur ZPO und StPO ab. Jeder Erwartungswert unten ist von Hand nachgerechnet.
"""
from datetime import date

import pytest

from app.services.deadline_calculator import (
    FRIST_CALCULATORS,
    anfechtung_146_inso,
    anmeldefrist_28_inso,
    ausschlussfrist_189_inso,
    insolvenzgeld_324_sgb3,
    kuendigungsfrist_113_inso,
    sofortige_beschwerde_6_inso,
    widerspruch_gegen_tabelle,
)

BY = "BY"


def test_anmeldefrist_default_six_weeks():
    # Montag, 10.03.2025 + 42 Tage = 21.04.2025 (Ostermontag) -> 22.04.
    r = anmeldefrist_28_inso(date(2025, 3, 10), bundesland=BY)
    assert r.raw_deadline == date(2025, 4, 21)
    assert r.deadline == date(2025, 4, 22)
    assert r.adjusted_for_holiday is True


def test_anmeldefrist_respects_statutory_bounds():
    # § 28 Abs. 1 S. 2 InsO: mindestens zwei Wochen, hoechstens drei Monate
    assert anmeldefrist_28_inso(date(2025, 3, 10), weeks=2, bundesland=BY)
    assert anmeldefrist_28_inso(date(2025, 3, 10), weeks=13, bundesland=BY)
    with pytest.raises(ValueError):
        anmeldefrist_28_inso(date(2025, 3, 10), weeks=1, bundesland=BY)
    with pytest.raises(ValueError):
        anmeldefrist_28_inso(date(2025, 3, 10), weeks=14, bundesland=BY)


def test_ausschlussfrist_189_is_two_weeks():
    r = ausschlussfrist_189_inso(date(2025, 5, 5), bundesland=BY)
    assert r.raw_deadline == date(2025, 5, 19)
    assert "Ausschlussfrist" in r.basis


def test_sofortige_beschwerde_two_weeks():
    r = sofortige_beschwerde_6_inso(date(2025, 5, 5), bundesland=BY)
    assert r.raw_deadline == date(2025, 5, 19)


def test_widerspruch_default_two_weeks():
    r = widerspruch_gegen_tabelle(date(2025, 5, 5), bundesland=BY)
    assert r.raw_deadline == date(2025, 5, 19)


def test_anfechtung_starts_at_end_of_year_not_at_opening():
    # § 146 InsO i. V. m. §§ 195, 199 BGB: die regelmaessige Verjaehrung
    # beginnt mit SCHLUSS DES JAHRES. Ab dem Eroeffnungstag zu rechnen waere
    # ein Fehler von bis zu zwoelf Monaten zulasten der Masse.
    r = anfechtung_146_inso(date(2025, 3, 10), bundesland=BY)
    assert r.raw_deadline == date(2028, 12, 31)
    # 01.01. ist Feiertag -> § 193 BGB schiebt auf den naechsten Werktag
    assert r.deadline == date(2029, 1, 2)


def test_anfechtung_is_independent_of_day_within_year():
    ends = {
        anfechtung_146_inso(date(2025, m, 1), bundesland=BY).raw_deadline
        for m in (1, 6, 12)
    }
    assert ends == {date(2028, 12, 31)}


def test_kuendigung_113_three_months_to_month_end():
    # § 113 S. 2 InsO: drei Monate zum Monatsende
    r = kuendigungsfrist_113_inso(date(2025, 3, 10), bundesland=BY)
    assert r.raw_deadline == date(2025, 6, 30)


def test_kuendigung_113_from_month_end():
    r = kuendigungsfrist_113_inso(date(2025, 1, 31), bundesland=BY)
    # 31.01. + 3 Monate = 30.04. (kein 31.04.), Monatsende bleibt 30.04.
    assert r.raw_deadline == date(2025, 4, 30)


def test_insolvenzgeld_two_months():
    r = insolvenzgeld_324_sgb3(date(2025, 3, 10), bundesland=BY)
    assert r.raw_deadline == date(2025, 5, 10)
    assert "Ausschlussfrist" in r.basis


def test_inso_calculators_are_registered():
    for key in (
        "anmeldefrist_28_inso", "widerspruch_tabelle_inso", "ausschlussfrist_189_inso",
        "sofortige_beschwerde_inso", "anfechtung_146_inso", "kuendigung_113_inso",
        "insolvenzgeld_324_sgb3",
    ):
        assert key in FRIST_CALCULATORS


def test_existing_zpo_calculators_still_registered():
    # Die Ergaenzung darf die bestehende Registry nicht ersetzen
    for key in ("berufung_einlegung", "einspruch_versaeumnisurteil", "wiedereinsetzung_stpo"):
        assert key in FRIST_CALCULATORS


def test_every_result_carries_its_legal_basis():
    # Nachvollziehbarkeit: jede Frist nennt ihre Norm
    results = [
        anmeldefrist_28_inso(date(2025, 3, 10), bundesland=BY),
        ausschlussfrist_189_inso(date(2025, 3, 10), bundesland=BY),
        anfechtung_146_inso(date(2025, 3, 10), bundesland=BY),
        kuendigungsfrist_113_inso(date(2025, 3, 10), bundesland=BY),
        insolvenzgeld_324_sgb3(date(2025, 3, 10), bundesland=BY),
    ]
    for r in results:
        assert "§" in r.basis
        assert r.note
