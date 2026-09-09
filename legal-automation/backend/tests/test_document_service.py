"""
Tests der Dokumentenverwaltung (reine Logik: Pruefsumme, Dateinamen,
Textextraktion aus PDF).
"""
import pytest

from app.services.document_service import (
    ALLOWED_CONTENT_TYPES,
    checksum_bytes,
    extract_text_from_pdf,
    safe_document_filename,
)


def test_checksum_is_stable_and_distinct():
    assert checksum_bytes(b"abc") == checksum_bytes(b"abc")
    assert checksum_bytes(b"abc") != checksum_bytes(b"abd")
    assert len(checksum_bytes(b"abc")) == 64


def test_checksum_detects_duplicate_scan():
    # Derselbe Scan zweimal hochgeladen -> gleiche Pruefsumme -> kein Duplikat
    data = b"%PDF-1.4 Beispielinhalt"
    assert checksum_bytes(data) == checksum_bytes(bytes(data))


@pytest.mark.parametrize(
    "raw,expected_ok",
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\x\\Akte.pdf", "Akte.pdf"),
        ("normal.pdf", "normal.pdf"),
        ("Beschluss Ä Ö Ü.pdf", "Beschluss Ä Ö Ü.pdf"),
    ],
)
def test_safe_filename_strips_paths(raw, expected_ok):
    assert safe_document_filename(raw) == expected_ok


def test_safe_filename_never_empty():
    assert safe_document_filename("") == "anhang.bin"
    assert safe_document_filename(None) == "anhang.bin"
    assert safe_document_filename("...") == "anhang.bin"


def test_allowed_types_cover_the_kanzlei_workflow():
    # PDF (beA/Gerichtspost) und Scans (Bilder) sind der Kern
    assert "application/pdf" in ALLOWED_CONTENT_TYPES
    assert "image/png" in ALLOWED_CONTENT_TYPES
    assert "image/jpeg" in ALLOWED_CONTENT_TYPES
    # Ausfuehrbare/aktive Formate bewusst NICHT
    assert "text/html" not in ALLOWED_CONTENT_TYPES
    assert "image/svg+xml" not in ALLOWED_CONTENT_TYPES
    assert "application/octet-stream" not in ALLOWED_CONTENT_TYPES


def _minimal_pdf_with_text(text: str) -> bytes:
    """Erzeugt ein PDF mit Textschicht ueber reportlab (bereits Abhaengigkeit)."""
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.drawString(72, 720, text)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_extract_text_from_pdf_reads_text_layer():
    pdf = _minimal_pdf_with_text("Insolvenzverfahren 1501 IN 123/25")
    text, pages = extract_text_from_pdf(pdf)
    assert pages == 1
    assert "1501 IN 123/25" in text


def test_extract_text_from_pdf_counts_pages():
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(3):
        c.drawString(72, 720, f"Seite {i + 1}")
        c.showPage()
    c.save()
    _, pages = extract_text_from_pdf(buf.getvalue())
    assert pages == 3


def test_extract_text_from_empty_pdf_returns_empty():
    # Ein Scan ohne Textschicht liefert nichts -> das ist das Signal, OCR zu starten
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.showPage()
    c.save()
    text, pages = extract_text_from_pdf(buf.getvalue())
    assert text == ""
    assert pages == 1
