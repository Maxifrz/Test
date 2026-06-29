"""Tests für die (reine) Datenaufbereitung des Vergütungsantrags."""
from decimal import Decimal

from app.services.insvv_calculator import calculate_insvv
from app.services.verguetungsantrag_pdf import build_antrag_rows


def test_rows_contain_key_positions():
    result = calculate_insvv(
        Decimal("50000"),
        zuschlaege=[("Betriebsfortführung", Decimal("0.5"))],
        auslagen=Decimal("250"),
        vat_rate=Decimal("0.19"),
    )
    rows = build_antrag_rows(result)
    labels = [r[0] for r in rows]
    assert any("Berechnungsgrundlage" in l for l in labels)
    assert any("Regelvergütung" in l for l in labels)
    assert any("Betriebsfortführung" in l for l in labels)
    assert any("Auslagen" in l for l in labels)
    assert any("Umsatzsteuer" in l for l in labels)
    assert any("brutto" in l.lower() for l in labels)


def test_amounts_use_german_format():
    result = calculate_insvv(Decimal("50000"), vat_rate=Decimal("0"))
    rows = dict(build_antrag_rows(result))
    # Regelvergütung 16.250,00 € im deutschen Format
    assert rows["Regelvergütung (§ 2 InsVV)"] == "16.250,00 €"
