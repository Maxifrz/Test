"""
Tests für den InsVV-Vergütungsrechner gegen verifizierte Berechnungsbeispiele
(§ 2 Abs. 1 InsVV Staffel).
"""
from decimal import Decimal

import pytest

from app.services.insvv_calculator import (
    calculate_insvv,
    mindestverguetung,
    regelverguetung,
)


@pytest.mark.parametrize(
    "grundlage,expected",
    [
        (25_000, "10000.00"),    # 25k × 40%
        (50_000, "16250.00"),    # +25k × 25% = 6.250
        (100_000, "19750.00"),   # +50k × 7% = 3.500
        (250_000, "30250.00"),   # +200k × 7% = 14.000
        (500_000, "37750.00"),   # +250k × 3% = 7.500
        (1_000_000, "47750.00"), # +500k × 2% = 10.000
    ],
)
def test_regelverguetung_staffel(grundlage, expected):
    assert regelverguetung(Decimal(grundlage)) == Decimal(expected)


def test_regelverguetung_partial_first_bracket():
    # 10.000 € liegt in der ersten Stufe → 40%
    assert regelverguetung(Decimal("10000")) == Decimal("4000.00")


def test_regelverguetung_invalid():
    with pytest.raises(ValueError):
        regelverguetung(Decimal("0"))


def test_zuschlag_and_abschlag():
    # Regelvergütung 16.250 bei 50k; +50% Betriebsfortführung, -10% → netto +40%
    r = calculate_insvv(
        Decimal("50000"),
        zuschlaege=[("Betriebsfortführung", Decimal("0.5"))],
        abschlaege=[("vereinfachtes Verfahren", Decimal("0.1"))],
        vat_rate=Decimal("0"),
    )
    assert r.regelverguetung == Decimal("16250.00")
    # 16.250 × 1.4 = 22.750
    assert r.verguetung_nach_anpassung == Decimal("22750.00")
    assert len(r.adjustments) == 2
    assert r.adjustments[0].amount == Decimal("8125.00")   # +50%
    assert r.adjustments[1].amount == Decimal("-1625.00")  # -10%


def test_mindestverguetung_floor_applies_for_small_masse():
    # Sehr kleine Masse → Regelvergütung unter Mindestvergütung
    r = calculate_insvv(Decimal("1000"), anzahl_glaeubiger=3, vat_rate=Decimal("0"))
    # 1.000 × 40% = 400 < 1.400 Mindest → angehoben
    assert r.regelverguetung == Decimal("400.00")
    assert r.mindestverguetung_angewandt is True
    assert r.verguetung_nach_anpassung == Decimal("1400.00")


def test_mindestverguetung_override():
    r = calculate_insvv(
        Decimal("1000"), mindestverguetung_override=Decimal("2000.00"), vat_rate=Decimal("0")
    )
    assert r.verguetung_nach_anpassung == Decimal("2000.00")
    assert r.mindestverguetung_angewandt is True


def test_mindestverguetung_glaeubiger_staffel():
    assert mindestverguetung(5) == Decimal("1400.00")
    assert mindestverguetung(6) == Decimal("1550.00")   # +1 Stufe
    assert mindestverguetung(10) == Decimal("1550.00")  # noch in 1. Stufe (6-10)
    assert mindestverguetung(11) == Decimal("1700.00")  # +2 Stufen


def test_auslagen_and_vat():
    r = calculate_insvv(
        Decimal("50000"),
        auslagen=Decimal("250.00"),
        vat_rate=Decimal("0.19"),
    )
    # 16.250 + 250 = 16.500 netto; USt 19% = 3.135; brutto 19.635
    assert r.netto == Decimal("16500.00")
    assert r.umsatzsteuer == Decimal("3135.00")
    assert r.brutto == Decimal("19635.00")
