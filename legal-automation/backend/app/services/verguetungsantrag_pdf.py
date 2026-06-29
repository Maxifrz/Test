"""
Vergütungsantrag (PDF) für das Insolvenzgericht auf Basis einer InsVV-Berechnung.

Die Datenaufbereitung (`build_antrag_rows`) ist reine, testbare Logik. Die
PDF-Erzeugung nutzt reportlab (pure-python) per Lazy-Import, damit das Modul
ohne installiertes reportlab importierbar bleibt.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.services.insvv_calculator import InsVVResult


@dataclass
class AntragContext:
    gericht: str | None
    aktenzeichen: str | None
    schuldner: str | None
    verwalter: str | None
    matter_number: str | None


def _eur(v: Decimal) -> str:
    s = f"{v:,.2f}"
    # deutsches Format: 1.234,56
    return s.replace(",", "X").replace(".", ",").replace("X", ".") + " €"


def build_antrag_rows(result: InsVVResult) -> list[tuple[str, str]]:
    """Aufschlüsselung als (Bezeichnung, Betrag)-Zeilen — reine Logik, testbar."""
    rows: list[tuple[str, str]] = []
    rows.append(("Berechnungsgrundlage (§ 1 InsVV)", _eur(result.berechnungsgrundlage)))
    rows.append(("Regelvergütung (§ 2 InsVV)", _eur(result.regelverguetung)))
    for adj in result.adjustments:
        pct = (adj.percent * 100).quantize(Decimal("0.01"))
        sign = "+" if adj.percent >= 0 else "−"
        rows.append((f"  {adj.name} ({sign}{abs(pct)} %)", _eur(adj.amount)))
    rows.append(("Vergütung nach Anpassung (§ 3 InsVV)", _eur(result.verguetung_nach_anpassung)))
    if result.mindestverguetung_angewandt:
        rows.append(("  davon Mindestvergütung (§ 2 Abs. 2)", _eur(result.mindestverguetung)))
    rows.append(("Auslagen (§ 8 InsVV)", _eur(result.auslagen)))
    rows.append(("Netto", _eur(result.netto)))
    rows.append((f"Umsatzsteuer ({(result.vat_rate * 100).quantize(Decimal('0.1'))} %)", _eur(result.umsatzsteuer)))
    rows.append(("Gesamtbetrag (brutto)", _eur(result.brutto)))
    return rows


def render_pdf(result: InsVVResult, ctx: AntragContext) -> bytes:
    """Erzeugt den Vergütungsantrag als PDF (reportlab)."""
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 25 * mm

    def line(text: str, *, size: int = 10, dy: float = 6 * mm, bold: bool = False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(25 * mm, y, text)
        y -= dy

    line(f"An das {ctx.gericht or 'Insolvenzgericht'}", size=10)
    line(f"— Insolvenzgericht —   Az.: {ctx.aktenzeichen or ctx.matter_number or '—'}", size=10, dy=10 * mm)

    line("Vergütungsantrag des Insolvenzverwalters", size=14, bold=True, dy=8 * mm)

    line(f"Schuldner: {ctx.schuldner or '—'}")
    line(f"Insolvenzverwalter: {ctx.verwalter or '—'}", dy=8 * mm)

    line("Berechnung der Vergütung nach der InsVV:", bold=True, dy=7 * mm)
    for label, amount in build_antrag_rows(result):
        c.setFont("Helvetica", 10)
        c.drawString(28 * mm, y, label)
        c.drawRightString(width - 25 * mm, y, amount)
        y -= 6 * mm

    y -= 6 * mm
    c.setFont("Helvetica", 10)
    antrag = (
        "Es wird beantragt, die Vergütung und die Auslagen des Insolvenzverwalters "
        f"auf insgesamt {_eur(result.brutto)} (brutto) festzusetzen."
    )
    # einfache Umbruch-Logik
    words = antrag.split()
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > 95:
            c.drawString(25 * mm, y, cur)
            y -= 5 * mm
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        c.drawString(25 * mm, y, cur)
    y -= 14 * mm

    c.setFont("Helvetica", 9)
    c.drawString(25 * mm, y, "Hinweis: Entwurf — Vergütungssätze (insb. InsVV-Staffel und Mindestvergütung)")
    y -= 4 * mm
    c.drawString(25 * mm, y, "sind vor Einreichung anwaltlich zu prüfen.")

    c.showPage()
    c.save()
    return buf.getvalue()
