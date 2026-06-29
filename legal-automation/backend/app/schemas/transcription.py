from datetime import date, datetime

from pydantic import BaseModel, field_validator

MEETING_TYPES = {
    "Besprechung", "Mandantengespräch", "Zeugenvernehmung", "Verhandlung", "Sonstiges",
}


class TranscriptionCreate(BaseModel):
    """Metadata accompanying the audio upload (the file comes as multipart)."""
    title: str
    meeting_type: str = "Besprechung"
    meeting_date: date
    matter_id: int | None = None

    @field_validator("meeting_type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        if v not in MEETING_TYPES:
            raise ValueError(f"meeting_type must be one of {sorted(MEETING_TYPES)}")
        return v


class SegmentResponse(BaseModel):
    id: int
    segment_index: int
    speaker: str
    speaker_label: str | None
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None
    edited: bool

    model_config = {"from_attributes": True}


class SegmentEdit(BaseModel):
    text: str | None = None
    speaker_label: str | None = None


class SpeakerRename(BaseModel):
    speaker: str  # the raw diarization label, e.g. "SPEAKER_00"
    label: str    # human label, e.g. "Richter"


class TranscriptionListItem(BaseModel):
    id: int
    title: str
    meeting_type: str
    meeting_date: date
    matter_id: int | None
    status: str
    progress_stage: str | None
    duration_seconds: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TranscriptionDetail(TranscriptionListItem):
    language: str
    model_used: str | None
    error_message: str | None
    original_filename: str | None
    segments: list[SegmentResponse] = []

    model_config = {"from_attributes": True}


class TranscriptionListResponse(BaseModel):
    items: list[TranscriptionListItem]
    total: int
    page: int
    page_size: int


class TranscriptionSearchHit(BaseModel):
    id: int
    title: str
    meeting_date: date
    matter_id: int | None
    snippet: str

    model_config = {"from_attributes": True}
