"""Tests für struktur-bewusstes Chunking (GraphRAG V3.0)."""
from app.ai.kri.chunking import chunk, chunk_generic, chunk_gesetz, chunk_urteil

GESETZ = """§ 133 Anfechtung
(1) Eine Rechtshandlung, die der Schuldner vorgenommen hat, ist anfechtbar.
(2) Bei einem entgeltlichen Vertrag gilt Absatz 1 mit der Maßgabe.
§ 134 Unentgeltliche Leistung
(1) Eine unentgeltliche Leistung des Schuldners ist anfechtbar."""


def test_chunk_gesetz_splits_by_section_and_absatz():
    chunks = chunk_gesetz(GESETZ)
    headings = [c.heading for c in chunks]
    assert "§ 133 Abs. 1" in headings
    assert "§ 133 Abs. 2" in headings
    assert "§ 134 Abs. 1" in headings
    # Inhalt korrekt zugeordnet
    abs2 = next(c for c in chunks if c.heading == "§ 133 Abs. 2")
    assert "entgeltlichen Vertrag" in abs2.text


def test_chunk_gesetz_without_absatz():
    chunks = chunk_gesetz("§ 1 Geltungsbereich Dieses Gesetz gilt für alle.")
    assert len(chunks) == 1
    assert chunks[0].heading.startswith("§ 1")


def test_chunk_urteil_by_randnummer():
    urteil = """Rn. 1 Die Klage ist zulässig.
Rn. 2 Die Berufung hat teilweise Erfolg.
Rn. 3 Die Kostenentscheidung folgt aus § 92 ZPO."""
    chunks = chunk_urteil(urteil)
    assert [c.heading for c in chunks] == ["Rn. 1", "Rn. 2", "Rn. 3"]
    assert "Berufung" in chunks[1].text


def test_chunk_generic_windows_and_overlap():
    text = "\n\n".join(f"Absatz {i} " + "x" * 400 for i in range(6))
    chunks = chunk_generic(text, max_chars=1000, overlap=50)
    assert len(chunks) > 1
    assert all(c.heading is None for c in chunks)


def test_chunk_dispatch_and_empty():
    assert chunk("", "gesetz") == []
    assert chunk("   ", "urteil") == []
    assert len(chunk(GESETZ, "gesetz")) >= 3
    # unbekannter Typ → generic
    assert len(chunk("Ein einzelner Absatz.", "intern_akte")) == 1


def test_gesetz_without_section_falls_back():
    chunks = chunk_gesetz("Kein Paragraph hier, nur Fließtext.")
    assert len(chunks) == 1
    assert chunks[0].heading is None
