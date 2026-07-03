"""Tests für Grounded-Answer-Helfer (Prompt, [S#]-Extraktion, Mapping)."""
from app.ai.kri.answer import (
    build_prompt,
    extract_source_refs,
    is_refusal,
    refs_to_chunk_ids,
)
from app.ai.kri.retrieval import validate_grounded


def test_build_prompt_contains_rules_context_question():
    p = build_prompt("Wann ist § 133 InsO einschlägig?", "[S1] § 133 InsO\nText…")
    assert "KEINE_GRUNDLAGE" in p          # Ablehnungs-Anweisung
    assert "[S1] § 133 InsO" in p          # Kontext
    assert "Wann ist § 133 InsO" in p      # Frage
    assert "keine Rechtsberatung" in p     # RDG-Hinweis


def test_extract_single_and_multi_refs():
    ans = "Die Anfechtung setzt Vorsatz voraus [S1]. Dies bestätigt die Rechtsprechung [S2, S3]."
    assert extract_source_refs(ans) == [1, 2, 3]


def test_extract_dedup_and_order():
    ans = "A [S2]. B [S1]. C [S2]."
    assert extract_source_refs(ans) == [2, 1]


def test_extract_no_refs():
    assert extract_source_refs("Antwort ohne Belege.") == []


def test_is_refusal():
    assert is_refusal("KEINE_GRUNDLAGE") is True
    assert is_refusal("Leider: keine Grundlage vorhanden") is True
    assert is_refusal("Die Grundlage ist § 133 InsO [S1].") is False


def test_refs_to_chunk_ids_maps_and_flags_unknown():
    used = [101, 102, 103]
    assert refs_to_chunk_ids([1, 3], used) == [101, 103]
    # [S9] existiert nicht → -1 → validate_grounded schlägt fehl
    mapped = refs_to_chunk_ids([1, 9], used)
    assert mapped == [101, -1]
    assert validate_grounded(mapped, used) is False


def test_full_grounding_chain_positive():
    """Ende-zu-Ende der reinen Kette: Antwort → Refs → chunk_ids → grounded."""
    used = [11, 22, 33]
    ans = "Aussage A [S1]. Aussage B [S2]."
    mapped = refs_to_chunk_ids(extract_source_refs(ans), used)
    assert validate_grounded(mapped, used) is True


def test_full_grounding_chain_hallucinated_source():
    used = [11]
    ans = "Aussage [S1], weitere [S4]."  # S4 wurde nie geliefert
    mapped = refs_to_chunk_ids(extract_source_refs(ans), used)
    assert validate_grounded(mapped, used) is False
