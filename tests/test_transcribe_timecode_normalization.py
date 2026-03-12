from datetime import timedelta

from transcribe import Transcript, normalize_segment_transcript_timecodes


def test_normalize_segment_timecodes_offsets_relative_times() -> None:
    transcripts = [
        Transcript(start="0:00:05", text="hello", voice=1),
        Transcript(start="0:08:26", text="world", voice=1),
    ]

    normalized = normalize_segment_transcript_timecodes(
        transcripts, segment_start=timedelta(minutes=29)
    )

    assert normalized[0].start == "0:29:05"


def test_normalize_segment_timecodes_keeps_absolute_times() -> None:
    transcripts = [
        Transcript(start="0:29:05", text="hello", voice=1),
        Transcript(start="0:30:00", text="world", voice=1),
    ]

    normalized = normalize_segment_transcript_timecodes(
        transcripts, segment_start=timedelta(minutes=29)
    )

    assert normalized[0].start == "0:29:05"
