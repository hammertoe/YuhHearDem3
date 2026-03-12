from datetime import timedelta

from transcribe import parse_timecode_to_timedelta


def test_parse_timecode_accepts_colon_milliseconds() -> None:
    result = parse_timecode_to_timedelta("1:0:16:600")

    assert result == timedelta(hours=1, minutes=0, seconds=16, milliseconds=600)


def test_parse_timecode_accepts_dot_milliseconds() -> None:
    result = parse_timecode_to_timedelta("1:00:16.600")

    assert result == timedelta(hours=1, minutes=0, seconds=16, milliseconds=600)
