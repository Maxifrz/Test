"""Tests für den Diarisierung↔Transkript-Merge (reine Logik)."""
from app.services.transcript_merge import (
    RawSegment,
    SpeakerTurn,
    assign_speaker,
    full_text,
    merge,
)


def test_assign_speaker_max_overlap():
    seg = RawSegment(start=10.0, end=14.0, text="Hallo")
    turns = [
        SpeakerTurn("SPEAKER_00", 0.0, 11.0),   # overlaps 1s
        SpeakerTurn("SPEAKER_01", 11.0, 20.0),  # overlaps 3s → wins
    ]
    assert assign_speaker(seg, turns) == "SPEAKER_01"


def test_assign_speaker_default_when_no_overlap():
    seg = RawSegment(start=100.0, end=101.0, text="x")
    turns = [SpeakerTurn("SPEAKER_00", 0.0, 5.0)]
    assert assign_speaker(seg, turns) == "SPEAKER_00"


def test_merge_orders_by_start_and_indexes():
    segs = [
        RawSegment(start=5.0, end=6.0, text="zweiter"),
        RawSegment(start=0.0, end=1.0, text="erster"),
    ]
    turns = [SpeakerTurn("SPEAKER_00", 0.0, 10.0)]
    merged = merge(segs, turns)
    assert [m.text for m in merged] == ["erster", "zweiter"]
    assert [m.segment_index for m in merged] == [0, 1]
    assert all(m.speaker == "SPEAKER_00" for m in merged)


def test_merge_strips_text():
    merged = merge([RawSegment(0.0, 1.0, "  hallo  ")], [])
    assert merged[0].text == "hallo"


def test_full_text_joins_segments():
    merged = merge(
        [RawSegment(0.0, 1.0, "Satz eins"), RawSegment(1.0, 2.0, "Satz zwei")],
        [SpeakerTurn("SPEAKER_00", 0.0, 5.0)],
    )
    assert full_text(merged) == "Satz eins\nSatz zwei"


def test_two_speakers_alternating():
    segs = [
        RawSegment(0.0, 2.0, "A spricht"),
        RawSegment(2.0, 4.0, "B spricht"),
    ]
    turns = [
        SpeakerTurn("SPEAKER_00", 0.0, 2.0),
        SpeakerTurn("SPEAKER_01", 2.0, 4.0),
    ]
    merged = merge(segs, turns)
    assert merged[0].speaker == "SPEAKER_00"
    assert merged[1].speaker == "SPEAKER_01"
