"""
Wrapper um faster-whisper (Transkription) und pyannote.audio (Diarisierung).

Alle ML-Bibliotheken werden LAZY importiert (innerhalb der Funktionen), damit
dieses Modul ohne installierte Modelle importierbar und der Rest der App
testbar bleibt. Es werden keine Netzwerk-Calls ausgelöst — Modelle müssen
lokal vorliegen (DSGVO: kein Datenabfluss).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from app.services.transcript_merge import RawSegment, SpeakerTurn


@dataclass
class TranscriptionOutput:
    segments: list[RawSegment]
    language: str
    duration: float
    model: str


def convert_to_wav(input_path: str, wav_path: str) -> None:
    """
    ffmpeg: beliebiges Audio/Video → WAV 16 kHz mono (Whisper-Eingabeformat).
    Läuft lokal, kein Netzwerk.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-vn",
            wav_path,
        ],
        check=True,
        capture_output=True,
    )


def transcribe(wav_path: str) -> TranscriptionOutput:
    """Run faster-whisper locally (German, word timestamps)."""
    from faster_whisper import WhisperModel

    from app.core.config import get_settings

    settings = get_settings()
    model = WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )
    segments_iter, info = model.transcribe(
        wav_path,
        language="de",
        word_timestamps=True,
        vad_filter=True,
    )
    segments = [
        RawSegment(
            start=float(s.start),
            end=float(s.end),
            text=s.text,
            confidence=getattr(s, "avg_logprob", None),
        )
        for s in segments_iter
    ]
    return TranscriptionOutput(
        segments=segments,
        language=info.language,
        duration=float(info.duration),
        model=settings.WHISPER_MODEL,
    )


def diarize(wav_path: str) -> list[SpeakerTurn]:
    """
    Run pyannote.audio speaker diarization locally. Returns speaker turns.
    Requires the pipeline weights to be present locally (no download at runtime).
    Returns an empty list if diarization is disabled.
    """
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.DIARIZATION_ENABLED:
        return []

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(settings.PYANNOTE_PIPELINE)
    diarization = pipeline(wav_path)

    turns: list[SpeakerTurn] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(speaker=speaker, start=float(turn.start), end=float(turn.end)))
    return turns
