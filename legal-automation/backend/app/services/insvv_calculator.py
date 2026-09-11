"""
InsVV-Vergütungsrechner (Insolvenzverwalter-Vergütung).

RECHTLICH KRITISCH — wie der RVG-Rechner: Fehler führen zu falschen
Vergütungsanträgen. Struktur spiegelt `rvg_calculator.py`.

Grundlagen:
- § 1 InsVV: Berechnungsgrundlage (Insolvenzmasse; Sonderregeln Aus-/Absonderung,
  Betriebsfortführung → Überschuss). Hier als bereits ermittelter Wert übergeben.
- § 2 Abs. 1 InsVV: degressive Regelvergütung (Staffel auf den Mehrbetrag).
- § 2 Abs. 2 InsVV: Mindestvergütung (seit 2021 angehoben).
- § 3 InsVV: Zu-/Abschläge (begründungspflichtig), prozentual auf die Regelvergütung.
- § 8 InsVV: Auslagen (konkret; Pauschsatz ist zeit-/jahresabhängig → Folgestufe).
- zzgl. USt.

KORREKTUR (wichtig): Bis einschließlich Version 3.0 rechnete dieses Modul mit
der Staffel VOR der InsVV-Reform 2021 (Schwellen 25.000/50.000/250.000/
500.000/25 Mio/50 Mio). Die Reform hat die Schwellen auf 35.000/70.000/
350.000/700.000/35 Mio/70 Mio angehoben; die alten Werte ergaben zu niedrige
Vergütungen (50.000 € Masse: 16.250 € statt 17.750 €). Die damaligen Tests
hatten die falschen Werte als "verifizierte Beispiele" festgeschrieben.

Ebenso § 2 Abs. 2: die Mindestvergütung von 1.400 € gilt bis zu 10 Gläubigern;
erst darüber erhöht sie sich je angefangene 5 Gläubiger um 150 €. Vorher stieg
sie hier bereits ab dem 6. Gläubiger.

VERIFIKATION VOR GO-LIVE: Die Zahlenwerte stehen gebündelt in `_STAFFEL` und
den MINDESTVERGUETUNG_*-Konstanten und sind vom verantwortlichen
Insolvenzverwalter gegen den geltenden Verordnungstext freizugeben. Die
Mindestvergütung lässt sich zusätzlich je Fall überschreiben
(`mindestverguetung_override`), etwa für das vereinfachte Verfahren (§ 13 InsVV).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
DEFAULT_VAT = Decimal("0.19")

# § 2 Abs. 1 InsVV in der Fassung seit der Reform 2021 —
# Staffel: (Obergrenze der Stufe oder None für "darüber", Satz auf den Mehrbetrag)
_STAFFEL: list[tuple[int | None, str]] = [
    (35_000, "0.40"),
    (70_000, "0.25"),
    (350_000, "0.07"),
    (700_000, "0.03"),
    (35_000_000, "0.02"),
    (70_000_000, "0.01"),
    (None, "0.005"),
]

# § 2 Abs. 2 InsVV — Mindestvergütung: 1.400 € bis 10 Gläubiger, darüber je
# angefangene 5 Gläubiger + 150 €.
MINDESTVERGUETUNG_BASIS = Decimal("1400.00")
MINDESTVERGUETUNG_STUFE_GLAEUBIGER = 5
MINDESTVERGUETUNG_STUFE_BETRAG = Decimal("150.00")
MINDESTVERGUETUNG_FREI_GLAEUBIGER = 10


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def regelverguetung(berechnungsgrundlage: Decimal) -> Decimal:
    """Regelvergütung nach § 2 Abs. 1 InsVV (degressive Staffel auf den Mehrbetrag)."""
    if berechnungsgrundlage <= 0:
        raise ValueError("Berechnungsgrundlage muss positiv sein")

    total = Decimal("0")
    lower = Decimal("0")
    for upper, rate_s in _STAFFEL:
        rate = Decimal(rate_s)
        if upper is None:
            total += (berechnungsgrundlage - lower) * rate
            break
        cap = Decimal(upper)
        if berechnungsgrundlage <= cap:
            total += (berechnungsgrundlage - lower) * rate
            return _money(total)
        total += (cap - lower) * rate
        lower = cap
    return _money(total)


def mindestverguetung(anzahl_glaeubiger: int) -> Decimal:
    """
    Mindestvergütung nach § 2 Abs. 2 InsVV: 1.400 € bis einschließlich 10
    Gläubiger, darüber je angefangene 5 Gläubiger + 150 €.
    (11–15 Gläubiger → 1.550 €, 16–20 → 1.700 € usw.)
    """
    extra = max(0, anzahl_glaeubiger - MINDESTVERGUETUNG_FREI_GLAEUBIGER)
    # je angefangene Stufe (ceil-Division)
    stufen = -(-extra // MINDESTVERGUETUNG_STUFE_GLAEUBIGER)
    return _money(MINDESTVERGUETUNG_BASIS + Decimal(stufen) * MINDESTVERGUETUNG_STUFE_BETRAG)


@dataclass
class FeeAdjustment:
    name: str
    percent: Decimal   # positiv = Zuschlag, negativ = Abschlag
    amount: Decimal    # vorzeichenbehaftet


@dataclass
class InsVVResult:
    berechnungsgrundlage: Decimal
    regelverguetung: Decimal
    adjustments: list[FeeAdjustment] = field(default_factory=list)
    verguetung_nach_anpassung: Decimal = Decimal("0.00")
    mindestverguetung: Decimal = Decimal("0.00")
    mindestverguetung_angewandt: bool = False
    auslagen: Decimal = Decimal("0.00")
    netto: Decimal = Decimal("0.00")
    umsatzsteuer: Decimal = Decimal("0.00")
    brutto: Decimal = Decimal("0.00")
    vat_rate: Decimal = DEFAULT_VAT


def calculate_insvv(
    berechnungsgrundlage: Decimal,
    *,
    zuschlaege: list[tuple[str, Decimal]] | None = None,
    abschlaege: list[tuple[str, Decimal]] | None = None,
    anzahl_glaeubiger: int = 1,
    auslagen: Decimal = Decimal("0"),
    vat_rate: Decimal = DEFAULT_VAT,
    mindestverguetung_override: Decimal | None = None,
) -> InsVVResult:
    """
    Vollständige InsVV-Vergütungsberechnung.

    zuschlaege/abschlaege: Listen von (Begründung, Prozentsatz als Decimal, z.B. 0.5 = 50%).
    Zu-/Abschläge wirken prozentual auf die Regelvergütung (§ 3 InsVV) und
    werden in einer Gesamtbetrachtung saldiert (herrschende Praxis).

    Übersteigt die Summe der Abschläge 100 %, wird bei 0 gekappt: eine negative
    Vergütung gibt es nicht. Die Mindestvergütung greift anschließend ohnehin.
    """
    regel = regelverguetung(berechnungsgrundlage)

    adjustments: list[FeeAdjustment] = []
    net_factor = Decimal("0")
    for name, pct in zuschlaege or []:
        adjustments.append(FeeAdjustment(name=name, percent=pct, amount=_money(regel * pct)))
        net_factor += pct
    for name, pct in abschlaege or []:
        adjustments.append(FeeAdjustment(name=name, percent=-pct, amount=_money(-(regel * pct))))
        net_factor -= pct

    verguetung = max(Decimal("0.00"), _money(regel + regel * net_factor))

    minverg = (
        mindestverguetung_override
        if mindestverguetung_override is not None
        else mindestverguetung(anzahl_glaeubiger)
    )
    applied = False
    if verguetung < minverg:
        verguetung = minverg
        applied = True

    auslagen = _money(auslagen)
    netto = verguetung + auslagen
    ust = _money(netto * vat_rate)
    brutto = netto + ust

    return InsVVResult(
        berechnungsgrundlage=berechnungsgrundlage,
        regelverguetung=regel,
        adjustments=adjustments,
        verguetung_nach_anpassung=verguetung,
        mindestverguetung=minverg,
        mindestverguetung_angewandt=applied,
        auslagen=auslagen,
        netto=_money(netto),
        umsatzsteuer=ust,
        brutto=_money(brutto),
        vat_rate=vat_rate,
    )
