"""
Tests des Tabellenauszugs nach § 178 Abs. 3 InsO.

Der Auszug ist fuer festgestellte Forderungen ein Vollstreckungstitel --
Betraege, Raenge und das Pruefungsergebnis muessen exakt stimmen.
"""
from datetime import date
from decimal import Decimal

from app.services.tabellenauszug_pdf import (
    TabellenauszugContext,
    TabellenRow,
    build_summary,
    build_tabellen_rows,
)


def rows():
    return [
        TabellenRow(1, "Muster GmbH", Decimal("12500.00"), Decimal("12500.00"),
                    "insolvenz_38", "festgestellt"),
        TabellenRow(2, "Beispiel AG", Decimal("8000.00"), None,
                    "insolvenz_38", "bestritten", "Hoehe der Forderung"),
        TabellenRow(3, "Gesellschafter", Decimal("50000.00"), Decimal("50000.00"),
                    "nachrangig_39_5", "festgestellt"),
    ]


def test_header_matches_insolvenztabelle():
    header = build_tabellen_rows(rows())[0]
    assert header == [
        "Lfd. Nr.", "Gläubiger", "angemeldet", "festgestellt", "Rang", "Ergebnis der Prüfung",
    ]


def test_rows_are_sorted_by_lfd_nr():
    unsorted = list(reversed(rows()))
    data = build_tabellen_rows(unsorted)[1:]
    assert [r[0] for r in data] == ["1", "2", "3"]


def test_amounts_use_german_format():
    data = build_tabellen_rows(rows())[1:]
    assert data[0][2] == "12.500,00 €"
    assert data[2][2] == "50.000,00 €"


def test_missing_established_amount_shows_dash():
    # Bestrittene Forderung ohne festgestellten Betrag darf keine 0,00 zeigen --
    # das waere eine inhaltlich falsche Aussage im Titel.
    data = build_tabellen_rows(rows())[1:]
    assert data[1][3] == "—"


def test_dispute_reason_is_carried_into_result_column():
    data = build_tabellen_rows(rows())[1:]
    assert data[1][5] == "bestritten (Hoehe der Forderung)"


def test_dispute_without_reason_stays_plain():
    r = [TabellenRow(1, "X", Decimal("100"), None, "insolvenz_38", "bestritten")]
    assert build_tabellen_rows(r)[1][5] == "bestritten"


def test_rank_labels_are_resolved():
    data = build_tabellen_rows(rows())[1:]
    assert data[0][4] == "§ 38 InsO"
    assert data[2][4] == "§ 39 I Nr. 5 InsO"


def test_unknown_rank_falls_back_to_raw_value():
    r = [TabellenRow(1, "X", Decimal("100"), None, "sonderfall", "angemeldet")]
    assert build_tabellen_rows(r)[1][4] == "sonderfall"


def test_summary_totals():
    summary = dict(build_summary(rows()))
    assert summary["Angemeldete Forderungen gesamt"] == "70.500,00 €"
    assert summary["Davon festgestellt"] == "62.500,00 €"
    assert summary["Davon bestritten"] == "8.000,00 €"
    assert summary["Anzahl Forderungen"] == "3"


def test_summary_for_empty_table():
    summary = dict(build_summary([]))
    assert summary["Angemeldete Forderungen gesamt"] == "0,00 €"
    assert summary["Anzahl Forderungen"] == "0"


def test_pdf_renders():
    from app.services.tabellenauszug_pdf import render_pdf

    ctx = TabellenauszugContext(
        gericht="AG München", aktenzeichen="1501 IN 123/25", schuldner="Muster GmbH",
        verwalter="RA Dr. Eckert", matter_number="2025-0007-MUS",
        pruefungstermin=date(2025, 6, 12), glaeubiger_name=None,
    )
    pdf = render_pdf(rows(), ctx)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_pdf_renders_for_single_creditor_extract():
    from app.services.tabellenauszug_pdf import render_pdf

    ctx = TabellenauszugContext(
        gericht=None, aktenzeichen=None, schuldner=None, verwalter=None,
        matter_number=None, pruefungstermin=None, glaeubiger_name="Muster GmbH",
    )
    assert render_pdf(rows()[:1], ctx).startswith(b"%PDF")
