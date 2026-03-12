from datetime import timedelta

from transcribe import build_caption_context, parse_vtt_cues


def test_parse_vtt_cues_extracts_entries() -> None:
    vtt_text = """WEBVTT

00:00:01.000 --> 00:00:03.000
This session is resumed.

00:00:05.000 --> 00:00:07.000
We return to the appropriation bill.
"""

    cues = parse_vtt_cues(vtt_text)

    assert len(cues) == 2
    assert cues[0][2] == "This session is resumed."


def test_build_caption_context_within_segment() -> None:
    cues = [
        (1.0, 3.0, "This session is resumed."),
        (5.0, 7.0, "We return to the appropriation bill."),
        (20.0, 22.0, "Later content."),
    ]

    context = build_caption_context(
        cues,
        segment_start=timedelta(seconds=4),
        segment_end=timedelta(seconds=10),
        buffer_seconds=0,
        max_chars=1000,
    )

    assert "appropriation bill" in context
    assert "Later content" not in context
