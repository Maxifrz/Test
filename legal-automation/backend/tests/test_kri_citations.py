"""Tests für die Zitations-Extraktion (GraphRAG V3.0)."""
from app.ai.kri.citations import extract_citations


def test_extract_norm_with_gesetz():
    cits = extract_citations("Die Anfechtung nach § 133 InsO setzt Vorsatz voraus.")
    norms = [c.normalized for c in cits if c.citation_type == "norm"]
    assert "§ 133 InsO" in norms


def test_extract_norm_with_absatz_and_art():
    cits = extract_citations("Vgl. § 133 Abs. 2 InsO sowie Art. 6 DSGVO.")
    norms = {c.normalized for c in cits if c.citation_type == "norm"}
    assert "§ 133 Abs. 2 InsO" in norms
    assert "Art. 6 DSGVO" in norms


def test_unknown_gesetz_abbrev_ignored():
    # "XYZ" ist kein bekanntes Kürzel → kein Norm-Treffer (Rauschreduktion)
    cits = extract_citations("Siehe § 5 XYZ irgendwo.")
    assert [c for c in cits if c.citation_type == "norm"] == []


def test_extract_aktenzeichen():
    cits = extract_citations("Urteil des BGH, Az. IX ZR 100/23, sowie 12 O 345/24.")
    az = {c.normalized for c in cits if c.citation_type == "aktenzeichen"}
    assert "IX ZR 100/23" in az
    assert "12 O 345/24" in az


def test_extract_ecli():
    cits = extract_citations("ECLI:DE:BGH:2023:120923UIXZR100.23.0 ist einschlägig.")
    ecli = [c for c in cits if c.citation_type == "ecli"]
    assert len(ecli) == 1
    assert ecli[0].normalized.startswith("ECLI:DE:BGH:2023")


def test_dedup():
    cits = extract_citations("§ 133 InsO ... nochmals § 133 InsO ...")
    norms = [c for c in cits if c.normalized == "§ 133 InsO"]
    assert len(norms) == 1


def test_no_citations():
    assert extract_citations("Ein Satz ohne jede Fundstelle.") == []
