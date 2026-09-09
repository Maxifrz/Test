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


# --- Anrechnung der Geschaeftsgebuehr (Vorbem. 3 Abs. 4 VV RVG) ---

def test_anrechnung_faktor_haelftig_mit_deckel():
    from app.services.rvg_calculator import anrechnung_faktor

    assert anrechnung_faktor(Decimal("1.3")) == Decimal("0.65")
    assert anrechnung_faktor(Decimal("1.5")) == Decimal("0.75")
    # Deckel bei 0,75
    assert anrechnung_faktor(Decimal("2.0")) == Decimal("0.75")
    assert anrechnung_faktor(Decimal("2.5")) == Decimal("0.75")


def test_geschaeftsgebuehr_wird_auf_verfahrensgebuehr_angerechnet():
    # 10.000 EUR Gegenstandswert -> 1,0-Gebuehr = 614 EUR
    r = calculate_rvg(
        Decimal("10000"),
        [("Geschäftsgebühr (VV 2300)", Decimal("1.3")),
         ("Verfahrensgebühr (VV 3100)", Decimal("1.3"))],
        vat_rate=Decimal("0"),
    )
    assert r.wertgebuehr_1_0 == Decimal("614")
    assert r.gebuehren_summe == Decimal("1596.40")     # 2 x 798,20
    assert r.anrechnung == Decimal("-399.10")          # 0,65 x 614
    assert r.anrechnung_hinweis is not None
    # 1.596,40 - 399,10 = 1.197,30 + 20 Auslagen
    assert r.netto == Decimal("1217.30")


def test_anrechnung_kann_abgeschaltet_werden():
    fees = [("Geschäftsgebühr (VV 2300)", Decimal("1.3")),
            ("Verfahrensgebühr (VV 3100)", Decimal("1.3"))]
    mit = calculate_rvg(Decimal("10000"), fees, vat_rate=Decimal("0"))
    ohne = calculate_rvg(Decimal("10000"), fees, vat_rate=Decimal("0"), anrechnung=False)
    assert ohne.anrechnung == Decimal("0.00")
    assert ohne.netto - mit.netto == Decimal("399.10")


def test_keine_anrechnung_ohne_verfahrensgebuehr():
    r = calculate_rvg(
        Decimal("10000"),
        [("Geschäftsgebühr (VV 2300)", Decimal("1.3"))],
        vat_rate=Decimal("0"),
    )
    assert r.anrechnung == Decimal("0.00")
    assert r.anrechnung_hinweis is None


def test_keine_anrechnung_ohne_geschaeftsgebuehr():
    r = calculate_rvg(
        Decimal("10000"),
        [("Verfahrensgebühr (VV 3100)", Decimal("1.3")),
         ("Terminsgebühr (VV 3104)", Decimal("1.2"))],
        vat_rate=Decimal("0"),
    )
    assert r.anrechnung == Decimal("0.00")


def test_anrechnung_uebersteigt_verfahrensgebuehr_nicht():
    # Hohe Geschaeftsgebuehr, niedrige Verfahrensgebuehr: der Abzug darf die
    # Verfahrensgebuehr nicht uebersteigen (sonst negative Position).
    r = calculate_rvg(
        Decimal("10000"),
        [("Geschäftsgebühr (VV 2300)", Decimal("2.5")),
         ("Verfahrensgebühr (VV 3100)", Decimal("0.5"))],
        vat_rate=Decimal("0"),
    )
    verfahrens = next(p for p in r.positions if "3100" in p.name)
    assert -r.anrechnung <= verfahrens.amount


def test_auslagenpauschale_bemisst_sich_nach_anrechnung():
    # Kleiner Wert: 20 % der Gebuehren liegen unter dem 20-EUR-Deckel, sodass
    # die Anrechnung die Pauschale tatsaechlich mindert.
    r = calculate_rvg(
        Decimal("500"),
        [("Geschäftsgebühr (VV 2300)", Decimal("1.3")),
         ("Verfahrensgebühr (VV 3100)", Decimal("1.3"))],
        vat_rate=Decimal("0"),
    )
    gebuehren_nach = r.gebuehren_summe + r.anrechnung
    assert r.auslagenpauschale == min(
        (gebuehren_nach * Decimal("0.20")).quantize(Decimal("0.01")), Decimal("20.00")
    )
