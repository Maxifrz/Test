"""
Tests der E-Mail-Korrekturen: echtes Sendedatum, Threading ueber References,
BCC bei Mehrfachempfaengern, Message-ID und Empfaenger-Chunking.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.services.email_service import (
    build_mime,
    chunk_recipients,
    parse_date_header,
    parse_raw_email,
    thread_root,
)

# --- Sendedatum aus dem Date-Header ---

def test_parse_date_header_with_timezone():
    dt = parse_date_header("Tue, 15 Oct 2024 09:30:00 +0200")
    assert dt == datetime(2024, 10, 15, 9, 30, tzinfo=timezone(timedelta(hours=2)))


def test_parse_date_header_without_timezone_assumes_utc():
    dt = parse_date_header("Tue, 15 Oct 2024 09:30:00")
    assert dt is not None and dt.tzinfo is not None


@pytest.mark.parametrize("raw", [None, "", "voelliger Unsinn", "32 Foo 9999"])
def test_parse_date_header_tolerates_broken_input(raw):
    assert parse_date_header(raw) is None


def test_parsed_email_carries_real_send_date():
    raw = (
        b"From: gericht@justiz.example\r\n"
        b"To: kanzlei@example.de\r\n"
        b"Subject: Ladung\r\n"
        b"Date: Tue, 15 Oct 2024 09:30:00 +0200\r\n"
        b"Message-ID: <abc@justiz.example>\r\n"
        b"\r\n"
        b"Terminmitteilung\r\n"
    )
    parsed = parse_raw_email(raw)
    # Nicht der Ingest-Zeitpunkt: aus dem Zugang laufen Fristen
    assert parsed["email_date"].year == 2024
    assert parsed["email_date"].month == 10
    assert parsed["email_date"].day == 15


def test_parsed_email_without_date_header_is_none():
    raw = b"From: a@b.de\r\nSubject: X\r\nMessage-ID: <x@y>\r\n\r\nText\r\n"
    assert parse_raw_email(raw)["email_date"] is None


# --- Threading ---

def test_thread_root_uses_first_reference():
    # RFC 5322: References fuehrt die Kette, aeltester Eintrag zuerst
    refs = "<root@a> <zweite@a> <dritte@a>"
    assert thread_root(refs, "<dritte@a>", "<vierte@a>") == "<root@a>"


def test_thread_root_falls_back_to_in_reply_to():
    assert thread_root(None, "<parent@a>", "<self@a>") == "<parent@a>"


def test_thread_root_falls_back_to_own_id():
    assert thread_root(None, None, "<self@a>") == "<self@a>"


def test_thread_root_ignores_malformed_references():
    assert thread_root("kein-winkel-klammern", "<parent@a>", "<self@a>") == "<parent@a>"


def test_deep_thread_stays_one_thread():
    # Der eigentliche Fehler: ab der dritten Ebene zerfiel ein Verlauf in
    # mehrere Threads, weil in_reply_to als thread_key diente.
    root = "<root@a>"
    keys = {
        thread_root(f"{root} <l2@a>", "<l2@a>", "<l3@a>"),
        thread_root(f"{root} <l2@a> <l3@a>", "<l3@a>", "<l4@a>"),
        thread_root(root, root, "<l2@a>"),
    }
    assert keys == {root}


# --- BCC ---

def test_multiple_recipients_go_to_bcc():
    mime = build_mime(
        from_name="Kanzlei", from_email="kanzlei@example.de",
        to_addresses=["a@x.de", "b@y.de", "c@z.de"],
        subject="Glaeubigerrundschreiben", body_text="Text", body_html=None,
        message_id="<m1@example.de>", use_bcc=True,
    )
    # Kein Empfaenger darf die Adressen der anderen im To-Header sehen
    assert "a@x.de" not in mime["To"]
    assert "b@y.de" not in mime["To"]
    assert "a@x.de" in mime["Bcc"] and "c@z.de" in mime["Bcc"]


def test_single_recipient_uses_to():
    mime = build_mime(
        from_name="Kanzlei", from_email="kanzlei@example.de",
        to_addresses=["mandant@example.de"],
        subject="Sachstand", body_text="Text", body_html=None,
        message_id="<m2@example.de>", use_bcc=False,
    )
    assert "mandant@example.de" in mime["To"]
    assert mime["Bcc"] is None


def test_message_id_is_set_on_the_wire():
    mime = build_mime(
        from_name="K", from_email="k@example.de", to_addresses=["a@x.de"],
        subject="S", body_text="T", body_html=None,
        message_id="<eindeutig@example.de>", use_bcc=False,
    )
    # Vorher stand die ID nur in der DB und nie in der Nachricht -> der
    # Datensatz war mit der realen Mail nicht korrelierbar
    assert mime["Message-ID"] == "<eindeutig@example.de>"


def test_reply_headers_are_set():
    mime = build_mime(
        from_name="K", from_email="k@example.de", to_addresses=["a@x.de"],
        subject="Re: S", body_text="T", body_html=None,
        message_id="<neu@example.de>", use_bcc=False,
        in_reply_to="<vorher@x.de>", references="<root@x.de> <vorher@x.de>",
    )
    assert mime["In-Reply-To"] == "<vorher@x.de>"
    assert mime["References"].startswith("<root@x.de>")


def test_html_alternative_is_attached():
    mime = build_mime(
        from_name="K", from_email="k@example.de", to_addresses=["a@x.de"],
        subject="S", body_text="Nur Text", body_html="<p>HTML</p>",
        message_id="<m@example.de>", use_bcc=False,
    )
    assert mime.is_multipart()


# --- Empfaenger-Chunking ---

def test_chunk_recipients_splits_evenly():
    addrs = [f"g{i}@example.de" for i in range(250)]
    chunks = chunk_recipients(addrs, 100)
    assert [len(c) for c in chunks] == [100, 100, 50]
    assert sum(len(c) for c in chunks) == 250


def test_chunk_recipients_single_chunk():
    assert chunk_recipients(["a@x.de", "b@x.de"], 100) == [["a@x.de", "b@x.de"]]


def test_chunk_recipients_empty():
    assert chunk_recipients([], 100) == []


def test_chunk_recipients_rejects_zero():
    with pytest.raises(ValueError):
        chunk_recipients(["a@x.de"], 0)
