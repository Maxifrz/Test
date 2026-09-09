"""Schemas der Dokumentenverwaltung."""
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, field_validator

from app.models.document import DOCUMENT_KINDS


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    matter_id: int
    title: str
    kind: str
    original_filename: str
    content_type: str | None
    size_bytes: int
    document_date: date | None
    ocr_status: str
    ocr_error: str | None
    page_count: int | None
    text_from_ocr: bool
    is_confidential: bool
    created_at: datetime | None


class DocumentDetail(DocumentResponse):
    extracted_text: str | None = None


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class DocumentUpdate(BaseModel):
    title: str | None = None
    kind: str | None = None
    document_date: date | None = None
    is_confidential: bool | None = None

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, v: str | None) -> str | None:
        if v is not None and v not in DOCUMENT_KINDS:
            raise ValueError(f"kind must be one of {sorted(DOCUMENT_KINDS)}")
        return v
