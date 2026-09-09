"""
Tests für den InsVV-Vergütungsrechner gegen die Staffel des § 2 Abs. 1 InsVV
in der Fassung SEIT DER REFORM 2021 (Schwellen 35.000/70.000/350.000/700.000).

Die frueheren Erwartungswerte in dieser Datei bildeten die Fassung VOR 2021 ab
und hatten damit einen Rechenfehler als "verifiziertes Beispiel" festgeschrieben.
Jeder Wert unten ist aus der Staffel von Hand nachgerechnet.
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
        (35_000, "14000.00"),     # 35k × 40%
        (50_000, "17750.00"),     # 14.000 + 15k × 25% = 3.750
        (70_000, "22750.00"),     # 14.000 + 35k × 25% = 8.750
        (100_000, "24850.00"),    # 22.750 + 30k × 7% = 2.100
        (350_000, "42350.00"),    # 22.750 + 280k × 7% = 19.600
        (700_000, "52850.00"),    # 42.350 + 350k × 3% = 10.500
        (1_000_000, "58850.00"),  # 52.850 + 300k × 2% = 6.000
    ],
)
def test_regelverguetung_staffel(grundlage, expected):
    assert regelverguetung(Decimal(grundlage)) == Decimal(expected)


def test_regelverguetung_partial_first_bracket():
    # 10.000 € liegt in der ersten Stufe (bis 35.000 €) → 40 %
    assert regelverguetung(Decimal("10000")) == Decimal("4000.00")


def test_regelverguetung_top_bracket():
    # Oberste Stufe: alles über 70 Mio. mit 0,5 %
    # 35 Mio → 14.000 + 35k×0,25 + 280k×0,07 + 350k×0,03 + 34,3 Mio×0,02
    unten = regelverguetung(Decimal("70000000"))
    assert regelverguetung(Decimal("70000000")) == unten
    # 10 Mio darüber → + 50.000
    assert regelverguetung(Decimal("80000000")) - unten == Decimal("50000.00")


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
    assert r.regelverguetung == Decimal("17750.00")
    # 17.750 × 1.4 = 24.850
    assert r.verguetung_nach_anpassung == Decimal("24850.00")
    assert len(r.adjustments) == 2
    assert r.adjustments[0].amount == Decimal("8875.00")   # +50%
    assert r.adjustments[1].amount == Decimal("-1775.00")  # -10%


def test_abschlaege_over_100_percent_never_go_negative():
    # Rechnerisch -20 %; eine negative Verguetung gibt es nicht, und die
    # Mindestverguetung greift ohnehin.
    r = calculate_insvv(
        Decimal("50000"),
        abschlaege=[("A", Decimal("0.7")), ("B", Decimal("0.5"))],
        vat_rate=Decimal("0"),
    )
    assert r.verguetung_nach_anpassung == Decimal("1400.00")
    assert r.mindestverguetung_angewandt is True


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
    # § 2 Abs. 2 InsVV: 1.400 € bis einschliesslich 10 Glaeubiger, darueber
    # je angefangene 5 Glaeubiger +150 €.
    assert mindestverguetung(1) == Decimal("1400.00")
    assert mindestverguetung(5) == Decimal("1400.00")
    assert mindestverguetung(10) == Decimal("1400.00")   # Grenze
    assert mindestverguetung(11) == Decimal("1550.00")   # +1 Stufe
    assert mindestverguetung(15) == Decimal("1550.00")   # noch 1. Stufe
    assert mindestverguetung(16) == Decimal("1700.00")   # +2 Stufen
    assert mindestverguetung(21) == Decimal("1850.00")   # +3 Stufen


def test_auslagen_and_vat():
    r = calculate_insvv(
        Decimal("50000"),
        auslagen=Decimal("250.00"),
        vat_rate=Decimal("0.19"),
    )
    # 17.750 + 250 = 18.000 netto; USt 19 % = 3.420; brutto 21.420
    assert r.netto == Decimal("18000.00")
    assert r.umsatzsteuer == Decimal("3420.00")
    assert r.brutto == Decimal("21420.00")
