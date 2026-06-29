"""
Tests für den RVG-Gebührenrechner gegen verifizierte Werte der Anlage 2
zu § 13 RVG (Fassung seit KostRÄG 2021).
"""
from decimal import Decimal

import pytest

from app.services.rvg_calculator import calculate_rvg, wertgebuehr


@pytest.mark.parametrize(
    "value,expected",
    [
        (500, "49.00"),
        (1_000, "88.00"),
        (1_500, "127.00"),
        (2_000, "166.00"),
        (3_000, "222.00"),
        (4_000, "278.00"),
        (5_000, "334.00"),
        (10_000, "614.00"),
        (25_000, "874.00"),
        (50_000, "1279.00"),
        (200_000, "2219.00"),
        (500_000, "3539.00"),
    ],
)
def test_wertgebuehr_table_values(value, expected):
    assert wertgebuehr(Decimal(value)) == Decimal(expected)


def test_wertgebuehr_rounds_up_to_next_tier():
    # 4.001 € fällt in die Stufe "bis 5.000" → 334,00 (angefangener Betrag)
    assert wertgebuehr(Decimal("4001")) == Decimal("334.00")
    # genau 500 € → unterste Gebühr
    assert wertgebuehr(Decimal("500")) == Decimal("49.00")
    # 501 € → nächste Stufe (bis 1.000) → 88,00
    assert wertgebuehr(Decimal("501")) == Decimal("88.00")


def test_wertgebuehr_over_500k():
    # 500.000 = 3.539,00; +1 angefangene 50.000er-Stufe (550.000) → +165
    assert wertgebuehr(Decimal("550000")) == Decimal("3704.00")
    assert wertgebuehr(Decimal("500001")) == Decimal("3704.00")


def test_wertgebuehr_invalid():
    with pytest.raises(ValueError):
        wertgebuehr(Decimal("0"))


def test_geschaeftsgebuehr_1_3_at_5000():
    """
    Gegenstandswert 5.000 €, Geschäftsgebühr 1,3 (VV 2300):
      1,0-Gebühr 334,00 × 1,3 = 434,20
      + Auslagenpauschale (VV 7002) gedeckelt auf 20,00
      Netto 454,20 + 19% USt 86,30 = Brutto 540,50
    """
    result = calculate_rvg(
        Decimal("5000"),
        [("Geschäftsgebühr (VV 2300)", Decimal("1.3"))],
    )
    assert result.wertgebuehr_1_0 == Decimal("334.00")
    assert result.positions[0].amount == Decimal("434.20")
    assert result.auslagenpauschale == Decimal("20.00")
    assert result.netto == Decimal("454.20")
    assert result.umsatzsteuer == Decimal("86.30")
    assert result.brutto == Decimal("540.50")


def test_auslagenpauschale_not_capped_for_small_fee():
    # Kleiner Gegenstandswert: 20% der Gebühr < 20 € → nicht gedeckelt
    result = calculate_rvg(Decimal("500"), [("Gebühr 1,0", Decimal("1.0"))])
    # 49,00 × 0,2 = 9,80
    assert result.auslagenpauschale == Decimal("9.80")
    assert result.netto == Decimal("58.80")


def test_multiple_fees_summed():
    result = calculate_rvg(
        Decimal("10000"),
        [
            ("Verfahrensgebühr (VV 3100)", Decimal("1.3")),
            ("Terminsgebühr (VV 3104)", Decimal("1.2")),
        ],
    )
    # 614,00 ×1,3 = 798,20 ; ×1,2 = 736,80 ; Summe 1.535,00
    assert result.positions[0].amount == Decimal("798.20")
    assert result.positions[1].amount == Decimal("736.80")
    assert result.gebuehren_summe == Decimal("1535.00")
    assert result.auslagenpauschale == Decimal("20.00")  # capped
    assert result.netto == Decimal("1555.00")


def test_vat_disabled_via_zero_rate():
    result = calculate_rvg(
        Decimal("5000"),
        [("Geschäftsgebühr", Decimal("1.3"))],
        vat_rate=Decimal("0"),
    )
    assert result.umsatzsteuer == Decimal("0.00")
    assert result.brutto == result.netto
