"""
Tests für die DSGVO-kritische Garantie: das Zwischen-WAV wird IMMER gelöscht,
auch wenn eine spätere Pipeline-Stufe fehlschlägt.
"""
import os

import pytest

from app.services.transcript_merge import RawSegment, SpeakerTurn
from app.services.transcription_pipeline import (
    PipelineResult,
    safe_delete,
    transcribe_and_merge,
)
from app.services.transcription_service import TranscriptionOutput


def _make_fakes(tmp_path, *, transcribe_raises=False):
    created = {}

    def fake_convert(input_path, wav_path):
        # Simulate ffmpeg producing the intermediate WAV
        with open(wav_path, "wb") as f:
            f.write(b"RIFF....fake wav")
        created["wav"] = wav_path

    def fake_transcribe(wav_path):
        assert os.path.exists(wav_path)  # WAV must exist during transcription
        if transcribe_raises:
            raise RuntimeError("whisper boom")
        return TranscriptionOutput(
            segments=[RawSegment(0.0, 1.0, "hallo", 0.9)],
            language="de",
            duration=1.0,
            model="fake",
        )

    def fake_diarize(wav_path):
        return [SpeakerTurn("SPEAKER_00", 0.0, 1.0)]

    return fake_convert, fake_transcribe, fake_diarize, created


def test_wav_deleted_on_success(tmp_path):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"audio")
    convert, transcribe, diarize, created = _make_fakes(tmp_path)

    result = transcribe_and_merge(
        str(audio), str(tmp_path),
        convert_fn=convert, transcribe_fn=transcribe, diarize_fn=diarize,
    )

    assert isinstance(result, PipelineResult)
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[0].text == "hallo"
    # DSGVO: intermediate WAV must be gone
    assert not os.path.exists(created["wav"])


def test_wav_deleted_even_when_transcription_fails(tmp_path):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"audio")
    convert, transcribe, diarize, created = _make_fakes(tmp_path, transcribe_raises=True)

    with pytest.raises(RuntimeError, match="whisper boom"):
        transcribe_and_merge(
            str(audio), str(tmp_path),
            convert_fn=convert, transcribe_fn=transcribe, diarize_fn=diarize,
        )

    # The finally block must have removed the WAV despite the exception
    assert not os.path.exists(created["wav"])


def test_safe_delete_is_idempotent(tmp_path):
    p = tmp_path / "nope.wav"
    safe_delete(str(p))  # does not raise on missing file
    p.write_bytes(b"x")
    safe_delete(str(p))
    assert not os.path.exists(p)
