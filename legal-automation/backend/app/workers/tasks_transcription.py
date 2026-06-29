"""Meeting transcription pipeline. Implemented in Phase 5."""
from app.workers.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="app.workers.tasks_transcription.process_transcription",
    max_retries=2,
)
def process_transcription(self, transcription_id: int):
    """10-stage transcription pipeline (Phase 5)."""
    # Stage placeholders — full implementation in Phase 5
    raise NotImplementedError("Transcription pipeline implemented in Phase 5")
