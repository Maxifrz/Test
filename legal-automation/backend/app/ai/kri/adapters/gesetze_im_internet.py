"""
Adapter: gesetze-im-internet.de (Bundesrecht als XML, "gii-norm"-Format).

Trennung wie überall im Projekt:
- parse_gii_xml / law_to_text: REINE Logik, gegen Fixtures getestet.
- fetch_gesetz: Netzwerk (httpx, ZIP) — läuft nur im echten Stack.

VOR PRODUKTIV-INGEST: Parser gegen einen echten Download verifizieren
(Format-Drift der DTD möglich); Nutzungsbedingungen der Quelle beachten
(Bundesrecht ist gemeinfrei, § 5 UrhG).
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree

BASE_URL = "https://www.gesetze-im-internet.de"


@dataclass
class ParsedNorm:
    enbez: str            # z. B. "§ 133"
    titel: str | None     # z. B. "Vorsätzliche Benachteiligung"
    text: str


@dataclass
class ParsedLaw:
    jurabk: str           # z. B. "InsO"
    langue: str           # Langtitel
    ausfertigung_datum: str | None
    norms: list[ParsedNorm] = field(default_factory=list)


_ENBEZ_NORM = re.compile(r"^(§+\s*\d+[a-z]?|Art\.?\s*\d+[a-z]?)", re.IGNORECASE)


def _text_of(elem) -> str:
    """Gesamten Text eines Elements extrahieren, Absätze (<P>) durch \\n trennen."""
    if elem is None:
        return ""
    parts: list[str] = []
    for p in elem.iter():
        tag = p.tag.upper() if isinstance(p.tag, str) else ""
        if tag == "P":
            t = "".join(p.itertext()).strip()
            if t:
                parts.append(t)
    if parts:
        return "\n".join(parts)
    return " ".join(t.strip() for t in elem.itertext() if t.strip())


def parse_gii_xml(xml_bytes: bytes) -> ParsedLaw:
    """Parst eine gii-norm-XML (eine Datei = ein Gesetz, viele <norm>-Elemente)."""
    root = ElementTree.fromstring(xml_bytes)
    jurabk, langue, ausf = "", "", None
    norms: list[ParsedNorm] = []

    for norm in root.iter("norm"):
        meta = norm.find("metadaten")
        if meta is None:
            continue
        if not jurabk:
            j = meta.findtext("jurabk") or meta.findtext("amtabk")
            if j:
                jurabk = j.strip()
        if not langue:
            l = meta.findtext("langue") or meta.findtext("kurzue")
            if l:
                langue = l.strip()
        if ausf is None:
            a = meta.findtext("ausfertigung-datum")
            if a:
                ausf = a.strip()

        enbez = (meta.findtext("enbez") or "").strip()
        if not enbez or not _ENBEZ_NORM.match(enbez):
            continue  # Inhaltsübersicht, Gliederung, Anlagen etc. überspringen

        titel = (meta.findtext("titel") or "").strip() or None
        textdaten = norm.find("textdaten")
        content = textdaten.find(".//text") if textdaten is not None else None
        body = _text_of(content)
        if body:
            norms.append(ParsedNorm(enbez=enbez, titel=titel, text=body))

    return ParsedLaw(jurabk=jurabk, langue=langue, ausfertigung_datum=ausf, norms=norms)


def law_to_text(law: ParsedLaw) -> str:
    """
    Serialisiert das Gesetz so, dass chunking.chunk_gesetz die §/Absatz-Struktur
    exakt wiederfindet: "§ 133 Titel\\n(1) …\\n(2) …".
    """
    blocks: list[str] = []
    for n in law.norms:
        head = f"{n.enbez} {n.titel}" if n.titel else n.enbez
        blocks.append(f"{head}\n{n.text}")
    return "\n".join(blocks)


def fetch_gesetz(abbrev: str, http_get=None) -> ParsedLaw:
    """
    Lädt <BASE_URL>/<abbrev>/xml.zip und parst die enthaltene XML.
    `http_get` injizierbar (Tests/Alternativ-Client); Default: httpx.
    """
    slug = abbrev.strip().lower()
    if not re.fullmatch(r"[a-z0-9_\-]{1,40}", slug):
        raise ValueError(f"Ungültiges Gesetzeskürzel: {abbrev!r}")
    url = f"{BASE_URL}/{slug}/xml.zip"

    if http_get is None:
        import httpx

        def http_get(u: str) -> bytes:  # type: ignore[misc]
            resp = httpx.get(u, timeout=60.0, follow_redirects=True)
            resp.raise_for_status()
            return resp.content

    payload = http_get(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_names:
            raise ValueError(f"Kein XML in {url}")
        xml_bytes = zf.read(xml_names[0])
    return parse_gii_xml(xml_bytes)
