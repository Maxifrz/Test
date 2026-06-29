"""
Reine Logik zum Zusammenführen von Diarisierung (Sprecher-Turns) und
Transkript-Segmenten (Whisper). Keine ML-Abhängigkeiten → unit-testbar.

Diarisierung liefert: Liste von SpeakerTurn(speaker, start, end).
Whisper liefert:      Liste von Segment(start, end, text, confidence).

Jedem Transkript-Segment wird der Sprecher zugewiesen, dessen Turn die
größte zeitliche Überlappung mit dem Segment hat.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    confidence: float | None = None


@dataclass
class MergedSegment:
    segment_index: int
    speaker: str
    start: float
    end: float
    text: str
    confidence: float | None


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the overlap between [a_start,a_end] and [b_start,b_end] (>= 0)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(segment: RawSegment, turns: list[SpeakerTurn], default: str = "SPEAKER_00") -> str:
    """Return the speaker whose turn overlaps the segment most."""
    best_speaker = default
    best_overlap = 0.0
    for turn in turns:
        ov = _overlap(segment.start, segment.end, turn.start, turn.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = turn.speaker
    return best_speaker


def merge(segments: list[RawSegment], turns: list[SpeakerTurn]) -> list[MergedSegment]:
    """
    Combine Whisper segments with diarization turns. Consecutive segments from
    the same speaker are kept as separate entries (UI can group them); the
    speaker assignment is per-segment by maximum temporal overlap.
    """
    merged: list[MergedSegment] = []
    for idx, seg in enumerate(sorted(segments, key=lambda s: s.start)):
        speaker = assign_speaker(seg, turns)
        merged.append(
            MergedSegment(
                segment_index=idx,
                speaker=speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                confidence=seg.confidence,
            )
        )
    return merged


def full_text(segments: list[MergedSegment]) -> str:
    """Concatenate segment texts into a single FTS-indexable document."""
    return "\n".join(s.text for s in segments if s.text)
