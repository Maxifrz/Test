"""
Meeting-Transkriptions-Pipeline (Phase 5).

Stufen: Upload (bereits erfolgt) → Validierung → ffmpeg WAV → Diarisierung
→ Whisper → Merge → Ablage (Original Fernet-verschlüsselt) → DB-Persistenz
(Segmente + Volltext für FTS) → Status completed.

DSGVO: Zwischen-WAV wird in transcription_pipeline (finally) gelöscht; das
Original wird verschlüsselt als original.enc abgelegt und danach gelöscht;
kein Netzwerk-Call (Modelle lokal).
"""
import asyncio
import os

from app.workers.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="app.workers.tasks_transcription.process_transcription",
    max_retries=2,
)
def process_transcription(self, transcription_id: int):
    """Entry point invoked after the upload route enqueues a job."""
    return asyncio.run(_run(transcription_id))


async def _set_status(db, transcription, *, status=None, stage=None, error=None):
    if status is not None:
        transcription.status = status
    if stage is not None:
        transcription.progress_stage = stage
    if error is not None:
        transcription.error_message = error
    await db.commit()


async def _run(transcription_id: int) -> dict:
    from sqlalchemy import select

    from app.core.config import get_settings
    from app.core.deps import AsyncSessionLocal
    from app.core.encryption import encrypt_file
    from app.models.transcription import Transcription, TranscriptSegment
    from app.services.transcript_merge import full_text
    from app.services.transcription_pipeline import safe_delete, transcribe_and_merge

    settings = get_settings()

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Transcription).where(Transcription.id == transcription_id)
        )
        transcription = result.scalar_one_or_none()
        if transcription is None:
            return {"status": "error", "reason": "transcription not found"}

        storage_dir = transcription.storage_dir
        original_path = os.path.join(storage_dir, transcription.original_filename or "original")
        enc_path = os.path.join(storage_dir, "original.enc")

        await _set_status(db, transcription, status="processing", stage="validating")

        try:
            # Stages: convert → transcribe → diarize → merge (WAV auto-deleted in finally)
            await _set_status(db, transcription, stage="transcribing")
            pipeline = await asyncio.to_thread(
                transcribe_and_merge, original_path, storage_dir
            )

            # Persist segments + denormalized full text (FTS column maintained by trigger)
            await _set_status(db, transcription, stage="persisting")
            for seg in pipeline.segments:
                db.add(
                    TranscriptSegment(
                        transcription_id=transcription.id,
                        segment_index=seg.segment_index,
                        speaker=seg.speaker,
                        start_seconds=seg.start,
                        end_seconds=seg.end,
                        text=seg.text,
                        confidence=seg.confidence,
                    )
                )
            transcription.full_text = full_text(pipeline.segments)
            transcription.language = pipeline.language
            transcription.duration_seconds = pipeline.duration
            transcription.model_used = pipeline.model

            # Encrypt the original at rest, then remove the plaintext source
            await _set_status(db, transcription, stage="encrypting_original")
            await asyncio.to_thread(encrypt_file, original_path, enc_path)
            safe_delete(original_path)

            await _set_status(db, transcription, status="completed", stage="done")
            return {"status": "ok", "segments": len(pipeline.segments)}

        except Exception as exc:  # noqa: BLE001 — record failure, surface in UI
            await _set_status(
                db, transcription, status="failed", stage="error", error=str(exc)[:2000]
            )
            # Even on failure: never leave plaintext audio lying around
            safe_delete(original_path)
            return {"status": "error", "reason": str(exc)}
