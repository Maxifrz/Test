"""Tests für die Anhang-Extraktion in parse_raw_email (reine Logik, kein I/O)."""
from email.message import EmailMessage as PyEmailMessage

from app.services.email_service import (
    MAX_ATTACHMENT_BYTES,
    parse_raw_email,
    safe_attachment_filename,
)


def _build_mail(*, attachments: list[tuple[str, str, bytes]] = (), body="Sehr geehrte Damen und Herren") -> bytes:
    msg = PyEmailMessage()
    msg["From"] = "Amtsgericht Hannover <poststelle@ag-hannover.example>"
    msg["To"] = "kanzlei@eckert.example"
    msg["Subject"] = "Ladung im Verfahren 902 IN 123/24"
    msg["Message-ID"] = "<test-attach-1@example>"
    msg.set_content(body)
    for filename, ctype, payload in attachments:
        maintype, _, subtype = ctype.partition("/")
        msg.add_attachment(payload, maintype=maintype, subtype=subtype, filename=filename)
    return msg.as_bytes()


def test_attachment_extracted_with_metadata():
    pdf = b"%PDF-1.4 fake ladung " + b"x" * 100
    parsed = parse_raw_email(_build_mail(attachments=[("Ladung_902_IN_123-24.pdf", "application/pdf", pdf)]))

    assert parsed["body_text"].startswith("Sehr geehrte")
    assert len(parsed["attachments"]) == 1
    att = parsed["attachments"][0]
    assert att["filename"] == "Ladung_902_IN_123-24.pdf"
    assert att["content_type"] == "application/pdf"
    assert att["payload"] == pdf


def test_multiple_attachments_and_body_intact():
    parsed = parse_raw_email(
        _build_mail(
            attachments=[
                ("anlage1.pdf", "application/pdf", b"a" * 50),
                ("tabelle.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", b"b" * 50),
            ]
        )
    )
    assert [a["filename"] for a in parsed["attachments"]] == ["anlage1.pdf", "tabelle.xlsx"]
    assert parsed["body_text"]  # Body wird nicht als Anhang verschluckt


def test_oversize_attachment_skipped():
    big = b"x" * (MAX_ATTACHMENT_BYTES + 1)
    parsed = parse_raw_email(_build_mail(attachments=[("riesig.bin", "application/octet-stream", big)]))
    assert parsed["attachments"] == []


def test_mail_without_attachments_has_empty_list():
    parsed = parse_raw_email(_build_mail())
    assert parsed["attachments"] == []


def test_filename_sanitization_path_traversal():
    assert safe_attachment_filename("../../etc/passwd") == "passwd"
    assert safe_attachment_filename("..\\..\\win\\cmd.exe") == "cmd.exe"
    assert safe_attachment_filename(None) == "anhang.bin"
    assert safe_attachment_filename("   ") == "anhang.bin"
    # Umlaute bleiben erhalten, Sonderzeichen werden ersetzt
    assert safe_attachment_filename("Vergütung §3 (Entwurf).pdf") == "Vergütung _3 (Entwurf).pdf"


def test_filename_length_capped_keeps_extension():
    long = "a" * 300 + ".pdf"
    out = safe_attachment_filename(long)
    assert len(out) <= 140
    assert out.endswith(".pdf")
