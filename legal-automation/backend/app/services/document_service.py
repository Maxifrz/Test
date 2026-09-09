"""
Dokumentendienst: verschlüsselte Ablage, Texterkennung, Volltextsuche.

Reihenfolge beim Upload:
  1. Datei prüfen (Größe, Typ) und Prüfsumme bilden — Doppel-Uploads desselben
     Scans erzeugen kein zweites Dokument.
  2. Verschlüsselt ablegen. Der Klartext landet NIE auf der Platte.
  3. Text extrahieren: zuerst die PDF-Textschicht (schnell, exakt), nur bei
     leerem Ergebnis OCR (langsam, fehleranfällig). Das läuft asynchron im
     Worker, damit ein 200-Seiten-Scan den Upload nicht blockiert.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document

logger = logging.getLogger("app.documents")

# Nur Formate, für die es einen sinnvollen Textpfad gibt.
ALLOWED_CONTENT_TYPES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tif",
    "text/plain": ".txt",
}


def checksum_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_document_filename(name: str | None) -> str:
    """Wie bei E-Mail-Anhängen: kein Pfad, keine Sonderzeichen, begrenzte Länge."""
    from app.services.email_service import safe_attachment_filename

    return safe_attachment_filename(name)


async def find_by_checksum(db: AsyncSession, matter_id: int, checksum: str) -> Document | None:
    result = await db.execute(
        select(Document).where(
            Document.matter_id == matter_id,
            Document.checksum == checksum,
            Document.deleted_at.is_(None),
        )
    )
    return result.scalars().first()


async def store_document(
    db: AsyncSession,
    *,
    matter_id: int,
    title: str,
    kind: str,
    filename: str,
    content_type: str | None,
    payload: bytes,
    uploaded_by_id: int,
    document_date=None,
    is_confidential: bool = False,
) -> tuple[Document, bool]:
    """
    Legt ein Dokument verschlüsselt ab. Gibt (Dokument, war_neu) zurück —
    ein bereits vorhandener Scan wird nicht dupliziert.
    """
    import asyncio

    from app.core.config import get_settings
    from app.core.encryption import encrypt_bytes

    settings = get_settings()
    digest = checksum_bytes(payload)

    existing = await find_by_checksum(db, matter_id, digest)
    if existing is not None:
        return existing, False

    safe_name = safe_document_filename(filename)
    target_dir = Path(settings.STORAGE_ROOT) / "documents" / str(matter_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    # Prüfsumme im Dateinamen: eindeutig, ohne die DB befragen zu müssen
    enc_path = target_dir / f"{digest[:16]}_{safe_name}.enc"

    token = await asyncio.to_thread(encrypt_bytes, payload)
    await asyncio.to_thread(enc_path.write_bytes, token)

    doc = Document(
        matter_id=matter_id,
        title=title.strip() or safe_name,
        kind=kind,
        original_filename=safe_name,
        content_type=content_type,
        size_bytes=len(payload),
        storage_path=str(enc_path),
        checksum=digest,
        document_date=document_date,
        uploaded_by_id=uploaded_by_id,
        is_confidential=is_confidential,
        ocr_status="pending",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc, True


async def read_document_bytes(doc: Document) -> bytes:
    import asyncio

    from app.core.encryption import decrypt_bytes

    token = await asyncio.to_thread(Path(doc.storage_path).read_bytes)
    return await asyncio.to_thread(decrypt_bytes, token)


# --- Textextraktion ---

def extract_text_from_pdf(data: bytes) -> tuple[str, int]:
    """
    Text aus der PDF-Textschicht. Gibt (Text, Seitenzahl) zurück.

    Digital erzeugte PDFs (Schriftsätze, Gerichtspost über beA) tragen bereits
    Text — den auszulesen ist um Größenordnungen schneller und exakter als OCR.
    Erst wenn hier nichts zurückkommt, lohnt der Scan-Pfad.
    """
    try:
        import pypdf
    except ImportError:  # pragma: no cover — Abhängigkeit optional
        return "", 0

    import io

    reader = pypdf.PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: S112
            # Eine defekte Seite darf nicht das ganze Dokument verhindern —
            # der Rest des Textes ist mehr wert als ein Abbruch.
            continue
    return "\n".join(parts).strip(), len(reader.pages)


def ocr_image_bytes(data: bytes, languages: str = "deu+eng") -> str:
    """
    OCR über ein Bild. Läuft vollständig lokal (Tesseract) — wie die
    Transkription: kein Datenabfluss.
    """
    try:
        import io

        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "OCR nicht verfügbar: pytesseract + Pillow installieren und "
            "tesseract-ocr im Image bereitstellen"
        ) from exc

    with Image.open(io.BytesIO(data)) as img:
        return pytesseract.image_to_string(img, lang=languages).strip()


def ocr_pdf_bytes(data: bytes, languages: str = "deu+eng", max_pages: int = 100) -> str:
    """
    OCR über ein gescanntes PDF. `max_pages` deckelt den Aufwand: ein
    versehentlich hochgeladener 2.000-Seiten-Scan darf den Worker nicht für
    Stunden belegen.
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OCR für PDF benötigt pdf2image + poppler-utils") from exc

    pages = convert_from_bytes(data, dpi=300, fmt="png")
    import io

    out = []
    for page in pages[:max_pages]:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        out.append(ocr_image_bytes(buf.getvalue(), languages))
    return "\n".join(out).strip()


async def process_document_text(db: AsyncSession, doc: Document) -> Document:
    """
    Extrahiert den Text eines Dokuments und schreibt ihn zurück.
    Fehler landen sichtbar in ocr_error statt still zu verschwinden.
    """
    import asyncio

    from app.core.config import get_settings

    settings = get_settings()
    doc.ocr_status = "running"
    await db.commit()

    try:
        payload = await read_document_bytes(doc)

        if doc.content_type == "text/plain":
            doc.extracted_text = payload.decode("utf-8", errors="replace")
            doc.text_from_ocr = False
        elif doc.content_type == "application/pdf":
            text, pages = await asyncio.to_thread(extract_text_from_pdf, payload)
            doc.page_count = pages or None
            if len(text) >= 40:
                # Textschicht vorhanden — kein OCR nötig
                doc.extracted_text = text
                doc.text_from_ocr = False
            elif settings.OCR_ENABLED:
                doc.extracted_text = await asyncio.to_thread(
                    ocr_pdf_bytes, payload, settings.OCR_LANGUAGES
                )
                doc.text_from_ocr = True
            else:
                doc.ocr_status = "skipped"
                await db.commit()
                return doc
        elif doc.content_type in ("image/png", "image/jpeg", "image/tiff"):
            if not settings.OCR_ENABLED:
                doc.ocr_status = "skipped"
                await db.commit()
                return doc
            doc.extracted_text = await asyncio.to_thread(
                ocr_image_bytes, payload, settings.OCR_LANGUAGES
            )
            doc.text_from_ocr = True
        else:
            doc.ocr_status = "skipped"
            await db.commit()
            return doc

        doc.ocr_status = "done"
        doc.ocr_error = None
    except Exception as exc:
        doc.ocr_status = "failed"
        doc.ocr_error = f"{type(exc).__name__}: {exc}"[:2000]
        logger.exception("Textextraktion fuer Dokument %s fehlgeschlagen", doc.id)

    await db.commit()
    await db.refresh(doc)
    return doc


# --- Suche ---

async def search_documents(
    db: AsyncSession,
    *,
    query: str | None,
    matter_ids: set[int] | None,
    kind: str | None = None,
    limit: int = 50,
) -> list[Document]:
    """
    Volltextsuche über die extrahierten Texte (deutsche Konfiguration).

    `matter_ids = None` bedeutet Admin (kein Filter); eine leere Menge bedeutet
    "keine Akten zugewiesen" und liefert nichts — die Aktentrennung gilt auch
    hier.
    """
    stmt = select(Document).where(Document.deleted_at.is_(None))

    if matter_ids is not None:
        if not matter_ids:
            return []
        stmt = stmt.where(Document.matter_id.in_(matter_ids))
    if kind:
        stmt = stmt.where(Document.kind == kind)

    if query and query.strip():
        ts = func.to_tsvector(
            sql_text("'german'"),
            func.coalesce(Document.title, "") + " " + func.coalesce(Document.extracted_text, ""),
        )
        tsq = func.plainto_tsquery(sql_text("'german'"), query.strip())
        stmt = stmt.where(ts.op("@@")(tsq)).order_by(func.ts_rank(ts, tsq).desc())
    else:
        stmt = stmt.order_by(Document.id.desc())

    result = await db.execute(stmt.limit(limit))
    return list(result.scalars().all())
