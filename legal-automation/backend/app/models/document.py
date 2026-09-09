"""
Dokumentenverwaltung: Scans, PDFs und Schriftsätze mit Volltextsuche.

Ohne dieses Modul waren eingescannte Dokumente tote Anhänge: sie ließen sich
ablegen und wieder herunterladen, aber nicht durchsuchen. Für eine Kanzlei ist
der Volltext über Handakten der eigentliche Hebel — genau dort liegt die
Information, die man später sucht.

Ablage: Fernet-verschlüsselt unter STORAGE_ROOT/documents/<matter_id>/, wie
die E-Mail-Anhänge. Der extrahierte Text steht im Klartext in der Datenbank,
weil die deutsche Volltextsuche (tsvector) sonst nicht funktioniert — die
Abwägung ist in docs/dsgvo/TOM.md dokumentiert.
"""
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Date,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin

if TYPE_CHECKING:
    from app.models.matter import Matter

# Fachliche Einordnung eines Dokuments in der Handakte
DOCUMENT_KINDS = {
    "schriftsatz",       # eigener oder gegnerischer Schriftsatz
    "gerichtspost",      # Beschluss, Verfügung, Ladung
    "vertrag",
    "buchhaltung",       # Rechnungen, Kontoauszüge
    "korrespondenz",
    "gutachten",
    "sonstiges",
}

# Verarbeitungszustand der Texterkennung
OCR_STATUS = {"pending", "running", "done", "skipped", "failed"}


class Document(Base, SoftDeleteMixin):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    matter_id: Mapped[int] = mapped_column(
        ForeignKey("matters.id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False, default="sonstiges", index=True)

    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Verschlüsselte Datei auf der Platte
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    # SHA-256 des Klartexts — erkennt Doppel-Uploads desselben Scans
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Posteingangsdatum (nicht Upload-Datum): daran hängen Fristen
    document_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    # --- Texterkennung ---
    ocr_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    ocr_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Extrahierter Text (aus der PDF-Textschicht oder per OCR)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # true, wenn der Text per OCR erzeugt wurde (und damit fehleranfällig ist)
    text_from_ocr: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    uploaded_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    is_confidential: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    matter: Mapped["Matter"] = relationship("Matter")

    def __repr__(self) -> str:
        return f"<Document id={self.id} matter={self.matter_id} title={self.title!r}>"
