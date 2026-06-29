from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin


class Transcription(Base, SoftDeleteMixin):
    """A transcribed meeting/hearing, filed against a matter."""

    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    meeting_type: Mapped[str] = mapped_column(String(50), nullable=False, default="Besprechung")
    # Besprechung | Mandantengespräch | Zeugenvernehmung | Verhandlung | Sonstiges
    meeting_date: Mapped[date] = mapped_column(Date, nullable=False)

    matter_id: Mapped[int | None] = mapped_column(ForeignKey("matters.id"), nullable=True, index=True)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    # queued | processing | completed | failed
    progress_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    language: Mapped[str] = mapped_column(String(10), nullable=False, default="de")
    model_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    original_filename: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Directory under storage/transcriptions/ (naming: YYYY-MM-DD_Mandant_Akte_Typ)
    storage_dir: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Denormalized full transcript text — the tsvector FTS column is added in the
    # migration (server-maintained via trigger), not mapped here.
    full_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    segments: Mapped[list["TranscriptSegment"]] = relationship(
        "TranscriptSegment",
        back_populates="transcription",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.start_seconds",
    )

    def __repr__(self) -> str:
        return f"<Transcription id={self.id} status={self.status} matter={self.matter_id}>"


class TranscriptSegment(Base, TimestampMixin):
    """One diarized + transcribed segment. Editable; edits tracked in transcript_edits."""

    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transcription_id: Mapped[int] = mapped_column(
        ForeignKey("transcriptions.id"), nullable=False, index=True
    )
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)

    speaker: Mapped[str] = mapped_column(String(50), nullable=False, default="SPEAKER_00")
    speaker_label: Mapped[str | None] = mapped_column(String(255), nullable=True)  # e.g. "Richter"

    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    edited: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    transcription: Mapped["Transcription"] = relationship("Transcription", back_populates="segments")


class TranscriptEdit(Base, TimestampMixin):
    """Audit trail of manual transcript corrections (who changed what, previous value)."""

    __tablename__ = "transcript_edits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("transcript_segments.id"), nullable=False, index=True)
    transcription_id: Mapped[int] = mapped_column(ForeignKey("transcriptions.id"), nullable=False, index=True)
    editor_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    field: Mapped[str] = mapped_column(String(50), nullable=False)  # text | speaker_label
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str | None] = mapped_column(Text, nullable=True)
