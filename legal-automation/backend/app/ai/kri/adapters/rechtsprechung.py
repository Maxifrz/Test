"""
Adapter: rechtsprechung-im-internet.de (Bundesgerichte als XML).

- parse_rii_xml / case_to_text: REINE Logik, gegen Fixtures getestet.
- fetch_toc / fetch_case: Netzwerk (httpx, ZIP) — nur im echten Stack.

Die Quelle veröffentlicht ein Inhaltsverzeichnis (rii-toc.xml) mit Links auf
je ein ZIP pro Entscheidung. VOR PRODUKTIV-INGEST gegen echte Downloads
verifizieren (Format-Drift möglich); amtliche Werke sind gemeinfrei (§ 5 UrhG).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree

TOC_URL = "https://www.rechtsprechung-im-internet.de/rii-toc.xml"

# Reihenfolge der Textteile im serialisierten Dokument
_TEXT_PARTS = ["leitsatz", "tenor", "tatbestand", "entscheidungsgruende", "gruende"]
_PART_LABELS = {
    "leitsatz": "Leitsatz",
    "tenor": "Tenor",
    "tatbestand": "Tatbestand",
    "entscheidungsgruende": "Entscheidungsgründe",
    "gruende": "Gründe",
}


@dataclass
class ParsedCase:
    gericht: str
    datum: str | None          # "YYYYMMDD" wie geliefert
    aktenzeichen: str | None
    ecli: str | None
    titel: str                 # kompakter Anzeigetitel
    text: str                  # serialisierter Volltext (mit Abschnitts-Labels)


def _clean(elem) -> str:
    if elem is None:
        return ""
    return "\n".join(t.strip() for t in elem.itertext() if t.strip())


def parse_rii_xml(xml_bytes: bytes) -> ParsedCase:
    """Parst eine RII-Entscheidungs-XML (<dokument>-Wurzel)."""
    root = ElementTree.fromstring(xml_bytes)

    gericht = (root.findtext(".//gertyp") or "").strip()
    gerort = (root.findtext(".//gerort") or "").strip()
    if gerort and gerort.lower() not in gericht.lower():
        gericht = f"{gericht} {gerort}".strip()
    datum = (root.findtext(".//entsch-datum") or "").strip() or None
    az = (root.findtext(".//aktenzeichen") or "").strip() or None
    ecli = (root.findtext(".//ecli") or "").strip() or None

    parts: list[str] = []
    for tag in _TEXT_PARTS:
        body = _clean(root.find(f".//{tag}"))
        if body:
            parts.append(f"{_PART_LABELS[tag]}\n{body}")

    datum_de = f"{datum[6:8]}.{datum[4:6]}.{datum[0:4]}" if datum and len(datum) == 8 else (datum or "?")
    titel = f"{gericht or 'Gericht'}, Urteil/Beschluss vom {datum_de}" + (f" — {az}" if az else "")

    return ParsedCase(
        gericht=gericht or "?", datum=datum, aktenzeichen=az, ecli=ecli,
        titel=titel, text="\n\n".join(parts),
    )


def case_to_text(case: ParsedCase) -> str:
    """Volltext für die Ingestion (chunk_urteil greift auf Rn.-Struktur, sonst generisch)."""
    return case.text


def parse_toc(xml_bytes: bytes) -> list[str]:
    """Extrahiert die Download-Links (ZIP je Entscheidung) aus rii-toc.xml."""
    root = ElementTree.fromstring(xml_bytes)
    links: list[str] = []
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        if link:
            links.append(link)
    return links


def fetch_toc(http_get=None) -> list[str]:
    if http_get is None:
        http_get = _default_get
    return parse_toc(http_get(TOC_URL))


def fetch_case(zip_url: str, http_get=None) -> ParsedCase:
    if http_get is None:
        http_get = _default_get
    payload = http_get(zip_url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_names:
            raise ValueError(f"Kein XML in {zip_url}")
        return parse_rii_xml(zf.read(xml_names[0]))


def _default_get(url: str) -> bytes:
    import httpx

    resp = httpx.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.content
