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

VERIFIKATION VOR GO-LIVE: Die § 2 Abs. 1 Staffel ist langjährig etabliert und
gegen veröffentlichte Beispiele getestet (50.000 € → 16.250 €; 100.000 € →
19.750 €; 250.000 € → 30.250 €; 500.000 € → 37.750 €). Die **Mindestvergütung
(§ 2 Abs. 2)** ist konfigurierbar (`mindestverguetung_override`) — der exakte
geltende Betrag und die Staffelung je Gläubigerzahl sind aus dem aktuellen
InsVV-Text zu übernehmen und vom Insolvenzverwalter freizugeben.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
DEFAULT_VAT = Decimal("0.19")

# § 2 Abs. 1 InsVV — Staffel: (Obergrenze der Stufe oder None für "darüber", Satz)
_STAFFEL: list[tuple[int | None, str]] = [
    (25_000, "0.40"),
    (50_000, "0.25"),
    (250_000, "0.07"),
    (500_000, "0.03"),
    (25_000_000, "0.02"),
    (50_000_000, "0.01"),
    (None, "0.005"),
]

# § 2 Abs. 2 InsVV — Mindestvergütung (seit 2021). Grundbetrag + Erhöhung je
# angefangene 5 Gläubiger über 5. Konfigurierbar / zu verifizieren.
MINDESTVERGUETUNG_BASIS = Decimal("1400.00")
MINDESTVERGUETUNG_STUFE_GLAEUBIGER = 5
MINDESTVERGUETUNG_STUFE_BETRAG = Decimal("150.00")
MINDESTVERGUETUNG_FREI_GLAEUBIGER = 5


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
    Mindestvergütung nach § 2 Abs. 2 InsVV (vereinfachte, konfigurierbare
    Staffelung — vor Go-Live gegen geltenden InsVV-Text verifizieren).
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
    Zu-/Abschläge wirken prozentual auf die Regelvergütung (§ 3 InsVV).
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

    verguetung = _money(regel + regel * net_factor)

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
