from datetime import datetime

from pydantic import BaseModel, field_validator

from app.models.legal_knowledge import SOURCE_TYPES


class KiQueryRequest(BaseModel):
    question: str
    matter_id: int | None = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Frage zu kurz")
        return v.strip()


class KiSource(BaseModel):
    marker: str
    chunk_id: int
    heading: str | None
    document_title: str
    source_type: str
    external_id: str | None
    url_or_ref: str | None


class KiQueryResponse(BaseModel):
    query_id: int | None
    answer: str
    grounded: bool
    sources: list[KiSource]
    model: str
    disclaimer: str = "KI-Entwurf — keine Rechtsberatung, anwaltliche Prüfung erforderlich."


class KiIngestRequest(BaseModel):
    source_type: str
    title: str
    text: str
    external_id: str | None = None
    jurisdiction: str | None = None
    url_or_ref: str | None = None
    matter_id: int | None = None

    @field_validator("source_type")
    @classmethod
    def valid_source_type(cls, v: str) -> str:
        if v not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("Dokumenttext zu kurz")
        return v


class KiIngestResponse(BaseModel):
    document_id: int | None
    num_chunks: int
    duplicate: bool


class KiDocumentResponse(BaseModel):
    id: int
    source_type: str
    external_id: str | None
    title: str
    jurisdiction: str | None
    matter_id: int | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class KiFeedbackRequest(BaseModel):
    feedback: str  # up | down
    note: str | None = None

    @field_validator("feedback")
    @classmethod
    def valid_feedback(cls, v: str) -> str:
        if v not in {"up", "down"}:
            raise ValueError("feedback must be 'up' or 'down'")
        return v


class KiStatusResponse(BaseModel):
    enabled: bool
    ollama_available: bool
    llm_model: str
    embed_model: str
    num_documents: int
    num_chunks: int
