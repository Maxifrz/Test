"""Tests für Hybrid-Ranking + Grounding-Regeln (GraphRAG V3.0)."""
from app.ai.kri.retrieval import (
    Candidate,
    ContextChunk,
    Weights,
    apply_graph_boost,
    build_context,
    hybrid_score,
    is_sufficient,
    rank,
    validate_grounded,
)


def test_hybrid_score_weighting():
    c = Candidate(chunk_id=1, document_id=1, vector_score=1.0, fts_score=0.0, graph_boost=0.0)
    assert hybrid_score(c, Weights(0.6, 0.3, 0.1)) == 0.6


def test_rank_orders_by_score_desc_deterministic():
    cands = [
        Candidate(1, 10, vector_score=0.2, fts_score=0.2),
        Candidate(2, 11, vector_score=0.9, fts_score=0.8),
        Candidate(3, 12, vector_score=0.5, fts_score=0.5),
    ]
    ranked = rank(cands, Weights(), top_k=2)
    assert [c.chunk_id for c in ranked] == [2, 3]


def test_graph_boost_promotes_cited_document():
    cands = [
        Candidate(1, 10, vector_score=0.5),
        Candidate(2, 20, vector_score=0.5),
    ]
    apply_graph_boost(cands, cited_document_ids={20}, boost=1.0)
    ranked = rank(cands, Weights(vector=0.6, fts=0.3, graph=0.4))
    assert ranked[0].chunk_id == 2  # das zitierte Dokument steigt auf


def test_is_sufficient_threshold():
    strong = rank([Candidate(1, 1, vector_score=0.9, fts_score=0.9)])
    weak = rank([Candidate(2, 2, vector_score=0.1, fts_score=0.1)])
    assert is_sufficient(strong, min_score=0.35) is True
    assert is_sufficient(weak, min_score=0.35) is False
    assert is_sufficient([], min_score=0.35) is False


def test_build_context_tags_and_used_ids():
    chunks = [
        ContextChunk(101, 1, "§ 133 InsO", "Anfechtungstext."),
        ContextChunk(102, 2, "Rn. 5", "Urteilstext."),
    ]
    ctx, used = build_context(chunks, max_chars=6000)
    assert "[S1]" in ctx and "[S2]" in ctx
    assert "§ 133 InsO" in ctx
    assert used == [101, 102]


def test_build_context_respects_max_chars():
    chunks = [ContextChunk(i, i, None, "x" * 500) for i in range(10)]
    ctx, used = build_context(chunks, max_chars=1200)
    assert len(used) < 10  # nicht alle passen rein


def test_validate_grounded():
    assert validate_grounded([101, 102], retrieved_ids=[101, 102, 103]) is True
    assert validate_grounded([101, 999], retrieved_ids=[101, 102]) is False  # 999 nicht abgerufen
    assert validate_grounded([], retrieved_ids=[101]) is False  # keine Quelle = nicht grounded
