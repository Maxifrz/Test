"""
Tabellenauszug nach § 178 Abs. 3 InsO.

Nach dem Prüfungstermin trägt der Urkundsbeamte das Ergebnis in die Tabelle
ein; der Verwalter erteilt den Gläubigern auf Verlangen einen Auszug. Für
festgestellte Forderungen wirkt die Eintragung nach § 178 Abs. 3 InsO wie ein
rechtskräftiges Urteil gegenüber dem Schuldner und allen Insolvenzgläubigern —
der Auszug ist damit ein Vollstreckungstitel und kein Beiblatt.

Aufbau wie in `verguetungsantrag_pdf.py`: die Datenaufbereitung ist reine,
testbare Logik; reportlab wird lazy importiert.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

# Statusbezeichnungen, wie sie in die Tabelle eingetragen werden
STATUS_LABELS = {
    "festgestellt": "festgestellt",
    "bestritten": "bestritten",
    "angemeldet": "angemeldet (noch nicht geprüft)",
    "geprueft": "geprüft",
}

RANK_LABELS = {
    "insolvenz_38": "§ 38 InsO",
    "nachrangig_39": "§ 39 InsO",
    "nachrangig_39_1": "§ 39 I Nr. 1 InsO",
    "nachrangig_39_2": "§ 39 I Nr. 2 InsO",
    "nachrangig_39_3": "§ 39 I Nr. 3 InsO",
    "nachrangig_39_4": "§ 39 I Nr. 4 InsO",
    "nachrangig_39_5": "§ 39 I Nr. 5 InsO",
    "absonderung": "Absonderung (§§ 49 ff. InsO)",
    "masseverbindlichkeit": "Masseverbindlichkeit (§§ 53 ff. InsO)",
}


@dataclass
class TabellenauszugContext:
    gericht: str | None
    aktenzeichen: str | None
    schuldner: str | None
    verwalter: str | None
    matter_number: str | None
    pruefungstermin: date | None = None
    # Auszug für einen einzelnen Gläubiger oder Gesamtauszug für das Gericht
    glaeubiger_name: str | None = None


@dataclass
class TabellenRow:
    lfd_nr: int
    creditor_name: str
    claim_amount: Decimal
    established_amount: Decimal | None
    rank: str
    status: str
    dispute_reason: str | None = None


def _eur(v: Decimal | None) -> str:
    if v is None:
        return "—"
    s = f"{Decimal(v):,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") + " €"


def build_tabellen_rows(claims: list[TabellenRow]) -> list[list[str]]:
    """
    Erzeugt die Tabellenzeilen als Strings — reine Logik, testbar.

    Spalten entsprechen der Insolvenztabelle: laufende Nummer, Gläubiger,
    angemeldeter Betrag, festgestellter Betrag, Rang, Ergebnis der Prüfung.
    """
    header = [
        "Lfd. Nr.", "Gläubiger", "angemeldet", "festgestellt", "Rang", "Ergebnis der Prüfung",
    ]
    rows: list[list[str]] = [header]
    for c in sorted(claims, key=lambda x: x.lfd_nr):
        ergebnis = STATUS_LABELS.get(c.status, c.status)
        if c.status == "bestritten" and c.dispute_reason:
            ergebnis = f"bestritten ({c.dispute_reason})"
        rows.append([
            str(c.lfd_nr),
            c.creditor_name,
            _eur(c.claim_amount),
            _eur(c.established_amount),
            RANK_LABELS.get(c.rank, c.rank),
            ergebnis,
        ])
    return rows


def build_summary(claims: list[TabellenRow]) -> list[tuple[str, str]]:
    """Zusammenfassung unter der Tabelle."""
    angemeldet = sum((Decimal(c.claim_amount) for c in claims), Decimal("0"))
    festgestellt = sum(
        (Decimal(c.established_amount) for c in claims if c.established_amount is not None),
        Decimal("0"),
    )
    bestritten = sum(
        (Decimal(c.claim_amount) for c in claims if c.status == "bestritten"), Decimal("0")
    )
    return [
        ("Angemeldete Forderungen gesamt", _eur(angemeldet)),
        ("Davon festgestellt", _eur(festgestellt)),
        ("Davon bestritten", _eur(bestritten)),
        ("Anzahl Forderungen", str(len(claims))),
    ]


def render_pdf(claims: list[TabellenRow], ctx: TabellenauszugContext) -> bytes:
    """
    Erzeugt den Tabellenauszug als PDF. reportlab wird lazy importiert, damit
    das Modul (und seine Tests) ohne die Abhängigkeit lauffähig bleiben.
    """
    import io

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    # Querformat: sechs Spalten mit Gläubigernamen passen im Hochformat nicht
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title="Tabellenauszug (§ 178 Abs. 3 InsO)",
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Auszug aus der Insolvenztabelle", styles["Title"]))
    story.append(Paragraph("§ 178 Abs. 3 InsO", styles["Normal"]))
    story.append(Spacer(1, 8 * mm))

    kopf = [
        ("Insolvenzgericht", ctx.gericht),
        ("Aktenzeichen", ctx.aktenzeichen),
        ("Schuldner", ctx.schuldner),
        ("Insolvenzverwalter", ctx.verwalter),
        ("Interne Akte", ctx.matter_number),
        ("Prüfungstermin", ctx.pruefungstermin.strftime("%d.%m.%Y") if ctx.pruefungstermin else None),
        ("Auszug für", ctx.glaeubiger_name),
    ]
    kopf_rows = [[k, v] for k, v in kopf if v]
    if kopf_rows:
        t = Table(kopf_rows, colWidths=[45 * mm, 120 * mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 8 * mm))

    data = build_tabellen_rows(claims)
    # Lange Gläubigernamen und Bestreitensgründe umbrechen lassen
    cell = styles["BodyText"].clone("cell")
    cell.fontSize = 8
    cell.leading = 10
    wrapped = [
        [Paragraph(str(v), cell) if i in (1, 5) else str(v) for i, v in enumerate(row)]
        for row in data[1:]
    ]
    table = Table(
        [data[0]] + wrapped,
        colWidths=[16 * mm, 70 * mm, 30 * mm, 30 * mm, 35 * mm, 60 * mm],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (2, 1), (3, -1), "RIGHT"),
        ("ALIGN", (0, 1), (0, -1), "RIGHT"),
    ]))
    story.append(table)
    story.append(Spacer(1, 8 * mm))

    summary = Table([[k, v] for k, v in build_summary(claims)], colWidths=[70 * mm, 40 * mm])
    summary.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
    ]))
    story.append(summary)
    story.append(Spacer(1, 6 * mm))

    hinweis = styles["BodyText"].clone("hinweis")
    hinweis.fontSize = 8
    hinweis.textColor = colors.grey
    story.append(Paragraph(
        "Für festgestellte Forderungen wirkt die Eintragung in die Tabelle nach "
        "§ 178 Abs. 3 InsO wie ein rechtskräftiges Urteil gegenüber dem Schuldner "
        "und allen Insolvenzgläubigern. Maßgeblich ist allein die bei Gericht "
        "geführte Tabelle; dieser Auszug gibt den Stand zum Ausstellungszeitpunkt wieder.",
        hinweis,
    ))

    doc.build(story)
    return buffer.getvalue()
