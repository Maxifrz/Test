"""
Tests für die Quellen-Adapter (gesetze-im-internet.de, rechtsprechung-im-internet.de).

Fixtures bilden die realen XML-Formate (gii-norm-DTD bzw. RII-<dokument>) nach.
Die Fetch-Funktionen (Netzwerk) werden über injizierte http_get-Fakes getestet.
"""
import io
import zipfile

from app.ai.kri.adapters.gesetze_im_internet import (
    fetch_gesetz,
    law_to_text,
    parse_gii_xml,
)
from app.ai.kri.adapters.rechtsprechung import (
    case_to_text,
    fetch_case,
    parse_rii_xml,
    parse_toc,
)
from app.ai.kri.chunking import chunk_gesetz

GII_XML = """<?xml version="1.0" encoding="UTF-8"?>
<dokumente builddate="20240101">
  <norm doknr="BJNR1">
    <metadaten>
      <jurabk>InsO</jurabk>
      <ausfertigung-datum manuell="ja">1994-10-05</ausfertigung-datum>
      <kurzue>Insolvenzordnung</kurzue>
      <langue>Insolvenzordnung</langue>
    </metadaten>
    <textdaten><text format="XML"><Content><P>Inhaltsverzeichnis...</P></Content></text></textdaten>
  </norm>
  <norm doknr="BJNR2">
    <metadaten>
      <jurabk>InsO</jurabk>
      <enbez>§ 133</enbez>
      <titel format="parat">Vorsätzliche Benachteiligung</titel>
    </metadaten>
    <textdaten>
      <text format="XML"><Content>
        <P>(1) Anfechtbar ist eine Rechtshandlung, die der Schuldner vorgenommen hat.</P>
        <P>(2) Bei einem entgeltlichen Vertrag gilt Absatz 1 entsprechend.</P>
      </Content></text>
    </textdaten>
  </norm>
  <norm doknr="BJNR3">
    <metadaten>
      <jurabk>InsO</jurabk>
      <enbez>Inhaltsübersicht</enbez>
    </metadaten>
    <textdaten><text format="XML"><Content><P>...</P></Content></text></textdaten>
  </norm>
  <norm doknr="BJNR4">
    <metadaten>
      <jurabk>InsO</jurabk>
      <enbez>§ 134</enbez>
      <titel>Unentgeltliche Leistung</titel>
    </metadaten>
    <textdaten>
      <text format="XML"><Content><P>(1) Anfechtbar ist eine unentgeltliche Leistung des Schuldners.</P></Content></text>
    </textdaten>
  </norm>
</dokumente>
""".encode("utf-8")

RII_XML = """<?xml version="1.0" encoding="UTF-8"?>
<dokument>
  <gertyp>BGH</gertyp>
  <gerort></gerort>
  <entsch-datum>20230912</entsch-datum>
  <aktenzeichen>IX ZR 100/23</aktenzeichen>
  <ecli>ECLI:DE:BGH:2023:120923UIXZR100.23.0</ecli>
  <leitsatz><p>Zur Anfechtung nach § 133 InsO bei kongruenter Deckung.</p></leitsatz>
  <tenor><p>Die Revision wird zurückgewiesen.</p></tenor>
  <gruende><p>Rn. 1 Die Klage ist zulässig.</p><p>Rn. 2 Die Anfechtung setzt Vorsatz voraus.</p></gruende>
</dokument>
""".encode("utf-8")


def _zip_bytes(name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, content)
    return buf.getvalue()


# --- gesetze-im-internet.de ---

def test_parse_gii_extracts_norms_and_meta():
    law = parse_gii_xml(GII_XML)
    assert law.jurabk == "InsO"
    assert law.langue == "Insolvenzordnung"
    assert law.ausfertigung_datum == "1994-10-05"
    # Inhaltsübersicht + Rahmen-Norm werden übersprungen
    assert [n.enbez for n in law.norms] == ["§ 133", "§ 134"]
    assert law.norms[0].titel == "Vorsätzliche Benachteiligung"
    assert "(2) Bei einem entgeltlichen Vertrag" in law.norms[0].text


def test_law_to_text_roundtrips_into_chunker():
    """Kernanforderung: serialisiertes Gesetz → chunk_gesetz findet §/Absatz exakt."""
    law = parse_gii_xml(GII_XML)
    chunks = chunk_gesetz(law_to_text(law))
    headings = [c.heading for c in chunks]
    assert "§ 133 Abs. 1" in headings
    assert "§ 133 Abs. 2" in headings
    assert "§ 134 Abs. 1" in headings
    abs2 = next(c for c in chunks if c.heading == "§ 133 Abs. 2")
    assert "entgeltlichen Vertrag" in abs2.text


def test_fetch_gesetz_with_injected_http():
    payload = _zip_bytes("BJNR_inso.xml", GII_XML)
    law = fetch_gesetz("inso", http_get=lambda url: payload)
    assert law.jurabk == "InsO"
    assert len(law.norms) == 2


def test_fetch_gesetz_rejects_bad_abbrev():
    import pytest
    with pytest.raises(ValueError):
        fetch_gesetz("../etc/passwd", http_get=lambda url: b"")


# --- rechtsprechung-im-internet.de ---

def test_parse_rii_extracts_fields_and_sections():
    case = parse_rii_xml(RII_XML)
    assert case.gericht == "BGH"
    assert case.datum == "20230912"
    assert case.aktenzeichen == "IX ZR 100/23"
    assert case.ecli.startswith("ECLI:DE:BGH:2023")
    assert "12.09.2023" in case.titel and "IX ZR 100/23" in case.titel
    text = case_to_text(case)
    assert "Leitsatz" in text and "Tenor" in text and "Gründe" in text
    assert "kongruenter Deckung" in text


def test_parse_toc_and_fetch_case():
    toc = b"""<?xml version="1.0"?><rss><channel>
      <item><link>https://example.org/a.zip</link></item>
      <item><link>https://example.org/b.zip</link></item>
    </channel></rss>"""
    links = parse_toc(toc)
    assert links == ["https://example.org/a.zip", "https://example.org/b.zip"]

    payload = _zip_bytes("case.xml", RII_XML)
    case = fetch_case("https://example.org/a.zip", http_get=lambda url: payload)
    assert case.aktenzeichen == "IX ZR 100/23"
