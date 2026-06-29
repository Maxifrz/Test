"""
Orchestrierung der Transkriptions-Pipeline — ohne Celery/DB-Abhängigkeit,
damit die DSGVO-kritische Garantie (Intermediate-WAV wird IMMER gelöscht)
unit-testbar ist.

Die Funktionen convert/transcribe/diarize werden injiziert (Default: die
echten Wrapper aus transcription_service), sodass Tests Fakes übergeben können.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from app.services.transcript_merge import MergedSegment, merge


def safe_delete(path: str) -> None:
    """Delete a file if it exists; never raises (used in finally blocks)."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@dataclass
class PipelineResult:
    segments: list[MergedSegment]
    language: str
    duration: float
    model: str


def transcribe_and_merge(
    audio_path: str,
    work_dir: str,
    *,
    convert_fn=None,
    transcribe_fn=None,
    diarize_fn=None,
) -> PipelineResult:
    """
    WAV-Konvertierung → Transkription → Diarisierung → Merge.

    DSGVO-KRITISCH: Das Zwischen-WAV wird im `finally`-Block gelöscht — auch
    wenn eine spätere Stufe eine Exception wirft. Das ist nicht optional.
    """
    # Lazy default wiring (keeps ML imports out of import time / tests)
    if convert_fn is None or transcribe_fn is None or diarize_fn is None:
        from app.services import transcription_service

        convert_fn = convert_fn or transcription_service.convert_to_wav
        transcribe_fn = transcribe_fn or transcription_service.transcribe
        diarize_fn = diarize_fn or transcription_service.diarize

    wav_path = os.path.join(work_dir, "audio_16k.wav")
    try:
        convert_fn(audio_path, wav_path)
        out = transcribe_fn(wav_path)
        turns = diarize_fn(wav_path)
        merged = merge(out.segments, turns)
        return PipelineResult(
            segments=merged,
            language=out.language,
            duration=out.duration,
            model=out.model,
        )
    finally:
        # Datensparsamkeit (Art. 5 DSGVO): kein unverschlüsseltes Audio bleibt liegen.
        safe_delete(wav_path)
