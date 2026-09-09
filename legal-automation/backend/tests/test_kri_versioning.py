"""
Tests der Korpus-Versionierung und der gebuendelten Embeddings.

Hintergrund: ohne Stilllegung der Vorfassung entsteht bei jeder
Gesetzesaenderung ein ZUSAETZLICHES aktives Dokument. Die Recherche findet
danach zwei widersprechende Textstaende derselben Norm und belegt beide als
gueltig -- fuer ein juristisches Rechercheprodukt der gefaehrlichste
Fehlermodus ueberhaupt.
"""
import asyncio

import pytest

from app.ai.kri.service import IngestResult, _checksum


def test_checksum_changes_with_text():
    a = _checksum("gesetz", "INSO", "Fassung A")
    b = _checksum("gesetz", "INSO", "Fassung B")
    assert a != b


def test_checksum_stable_for_same_input():
    assert _checksum("gesetz", "INSO", "X") == _checksum("gesetz", "INSO", "X")


def test_checksum_separates_sources():
    # Gleicher Text, andere Quelle -> anderes Dokument
    assert _checksum("gesetz", "INSO", "X") != _checksum("urteil", "INSO", "X")


def test_checksum_separates_external_ids():
    assert _checksum("gesetz", "INSO", "X") != _checksum("gesetz", "ZPO", "X")


def test_ingest_result_reports_supersession():
    r = IngestResult(document_id=5, num_chunks=3, duplicate=False, superseded_document_id=4)
    assert r.superseded_document_id == 4


def test_ingest_result_defaults_to_no_supersession():
    r = IngestResult(document_id=5, num_chunks=3, duplicate=False)
    assert r.superseded_document_id is None


# --- Batch-Embeddings ---

class _FakeResponse:
    status_code = 200

    def __init__(self, dim: int, marker: float):
        self._dim = dim
        self._marker = marker

    def json(self):
        return {"embedding": [self._marker] * self._dim}


class _FakeClient:
    """Zaehlt Anfragen und liefert je Text ein unterscheidbares Embedding."""

    def __init__(self):
        self.calls: list[str] = []

    async def post(self, url, json=None, **kwargs):
        text = json["prompt"]
        self.calls.append(text)
        await asyncio.sleep(0)  # Nebenlaeufigkeit erzwingen
        return _FakeResponse(4, float(len(text)))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_embed_many_preserves_order(monkeypatch):
    from app.ai.llm import ollama_client as mod

    fake = _FakeClient()
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: fake)

    client = mod.OllamaClient.__new__(mod.OllamaClient)
    client.base_url = "http://x"
    client.embed_model = "test"
    client.llm_model = "test"

    texts = ["a", "bb", "ccc", "dddd"]
    result = asyncio.run(client.embed_many(texts, concurrency=2))

    # Reihenfolge MUSS erhalten bleiben -- die Chunk-Zuordnung haengt daran
    assert [vec[0] for vec in result] == [1.0, 2.0, 3.0, 4.0]
    assert len(result) == len(texts)


def test_embed_many_empty_list_makes_no_calls(monkeypatch):
    from app.ai.llm import ollama_client as mod

    fake = _FakeClient()
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: fake)

    client = mod.OllamaClient.__new__(mod.OllamaClient)
    client.base_url = "http://x"
    client.embed_model = "test"
    client.llm_model = "test"

    assert asyncio.run(client.embed_many([])) == []
    assert fake.calls == []


def test_embed_many_uses_single_client(monkeypatch):
    from app.ai.llm import ollama_client as mod

    created = []

    def factory(**kwargs):
        c = _FakeClient()
        created.append(c)
        return c

    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)

    client = mod.OllamaClient.__new__(mod.OllamaClient)
    client.base_url = "http://x"
    client.embed_model = "test"
    client.llm_model = "test"

    asyncio.run(client.embed_many(["a", "b", "c"], concurrency=3))
    # Genau EIN Client fuer alle Texte (vorher: einer pro Chunk)
    assert len(created) == 1
    assert len(created[0].calls) == 3
