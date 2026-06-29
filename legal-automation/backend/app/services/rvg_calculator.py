"""
RVG-Gebührenrechner.

RECHTLICH KRITISCH: Fehler hier führen zu falscher Abrechnung. Die
Wertgebühr wird nach **§ 13 Abs. 1 RVG (Fassung seit KostRÄG 2021,
gültig ab 01.01.2021)** berechnet — als stufenweise Formel, nicht als
statische Tabelle, damit beliebige Gegenstandswerte korrekt sind.

Verifikation gegen Anlage 2 zu § 13 RVG (Auszug):
  500 € → 49,00 | 2.000 € → 166,00 | 5.000 € → 334,00 | 10.000 € → 614,00
  25.000 € → 874,00 | 50.000 € → 1.279,00 | 200.000 € → 2.219,00 |
  500.000 € → 3.539,00

Der verantwortliche Anwalt muss die Sätze vor Go-Live freigeben; die
RVG-Sätze werden periodisch durch den Gesetzgeber angepasst.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")

# Post- und Telekommunikationspauschale (VV 7002): 20 % der Gebühren, max. 20 €
AUSLAGEN_RATE = Decimal("0.20")
AUSLAGEN_CAP = Decimal("20.00")
DEFAULT_VAT = Decimal("0.19")  # Umsatzsteuer (VV 7008)

# Stufen nach § 13 Abs. 1 RVG: (Obergrenze, Schritt, Erhöhung je angefangenem Schritt)
_TIERS: list[tuple[int, int, int]] = [
    (2_000, 500, 39),
    (10_000, 1_000, 56),
    (25_000, 3_000, 52),
    (50_000, 5_000, 81),
    (200_000, 15_000, 94),
    (500_000, 30_000, 132),
]
_BASE = Decimal("49")          # Gebühr bis 500 €
_OVER_500K = (50_000, 165)     # über 500.000 €: je angefangene 50.000 € → +165 €

# Gängige VV-Gebührensätze (Name, Regelfaktor) — Faktor ist anpassbar
STANDARD_FEES: dict[str, tuple[str, str]] = {
    "geschaeftsgebuehr_2300": ("Geschäftsgebühr (VV 2300)", "1.3"),
    "verfahrensgebuehr_3100": ("Verfahrensgebühr (VV 3100)", "1.3"),
    "terminsgebuehr_3104": ("Terminsgebühr (VV 3104)", "1.2"),
    "einigungsgebuehr_1000": ("Einigungsgebühr (VV 1000)", "1.5"),
}


def wertgebuehr(gegenstandswert: Decimal) -> Decimal:
    """1,0-Wertgebühr nach § 13 Abs. 1 RVG für einen Gegenstandswert in Euro."""
    if gegenstandswert <= 0:
        raise ValueError("Gegenstandswert muss positiv sein")

    if gegenstandswert <= 500:
        return _BASE

    gebuehr = _BASE
    lower = Decimal(500)
    for upper, step, inc in _TIERS:
        if gegenstandswert > lower:
            span_top = min(gegenstandswert, Decimal(upper))
            steps = math.ceil((span_top - lower) / Decimal(step))
            gebuehr += Decimal(steps) * Decimal(inc)
            lower = Decimal(upper)
        if gegenstandswert <= upper:
            return gebuehr

    # über 500.000 €
    step, inc = _OVER_500K
    steps = math.ceil((gegenstandswert - Decimal(500_000)) / Decimal(step))
    gebuehr += Decimal(steps) * Decimal(inc)
    return gebuehr


@dataclass
class FeePosition:
    name: str
    factor: Decimal
    amount: Decimal


@dataclass
class RVGResult:
    gegenstandswert: Decimal
    wertgebuehr_1_0: Decimal
    positions: list[FeePosition] = field(default_factory=list)
    gebuehren_summe: Decimal = Decimal("0.00")
    auslagenpauschale: Decimal = Decimal("0.00")
    netto: Decimal = Decimal("0.00")
    umsatzsteuer: Decimal = Decimal("0.00")
    brutto: Decimal = Decimal("0.00")
    vat_rate: Decimal = DEFAULT_VAT


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def calculate_rvg(
    gegenstandswert: Decimal,
    fees: list[tuple[str, Decimal]],
    *,
    add_auslagenpauschale: bool = True,
    vat_rate: Decimal = DEFAULT_VAT,
) -> RVGResult:
    """
    Berechnet eine RVG-Kostennote.

    fees: Liste von (Bezeichnung, Faktor), z.B. [("Geschäftsgebühr (VV 2300)", Decimal("1.3"))].
    """
    base = wertgebuehr(gegenstandswert)
    positions: list[FeePosition] = []
    gebuehren_summe = Decimal("0.00")

    for name, factor in fees:
        amount = _money(base * factor)
        positions.append(FeePosition(name=name, factor=factor, amount=amount))
        gebuehren_summe += amount

    auslagen = Decimal("0.00")
    if add_auslagenpauschale and gebuehren_summe > 0:
        auslagen = min(_money(gebuehren_summe * AUSLAGEN_RATE), AUSLAGEN_CAP)

    netto = gebuehren_summe + auslagen
    ust = _money(netto * vat_rate)
    brutto = netto + ust

    return RVGResult(
        gegenstandswert=gegenstandswert,
        wertgebuehr_1_0=base,
        positions=positions,
        gebuehren_summe=_money(gebuehren_summe),
        auslagenpauschale=_money(auslagen),
        netto=_money(netto),
        umsatzsteuer=ust,
        brutto=_money(brutto),
        vat_rate=vat_rate,
    )
