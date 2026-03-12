from transcribe import Transcript, validate_transcript_against_captions


def test_guardrail_accepts_similar_text() -> None:
    transcripts = [
        Transcript(start="0:00:01", text="This session is resumed prior to suspension.", voice=1)
    ]
    result = validate_transcript_against_captions(
        transcripts,
        "This session is resumed prior to the suspension.",
        min_similarity=45.0,
        max_seconds=None,
    )

    assert result.status == "ok"


def test_guardrail_flags_mismatch() -> None:
    transcripts = [
        Transcript(
            start="0:00:01",
            text="Quantum entanglement drives photon decoherence in superconducting qubits.",
            voice=1,
        )
    ]
    result = validate_transcript_against_captions(
        transcripts,
        "This session is resumed. Prior to the suspension this chamber was debating.",
        min_similarity=45.0,
        max_seconds=None,
    )

    assert result.status == "mismatch"


def test_guardrail_handles_missing_captions() -> None:
    transcripts = [Transcript(start="0:00:01", text="Some transcript text", voice=1)]
    result = validate_transcript_against_captions(
        transcripts,
        "",
        min_similarity=45.0,
        max_seconds=None,
    )

    assert result.status == "no_captions"


def test_guardrail_handles_empty_transcript() -> None:
    transcripts: list[Transcript] = []
    result = validate_transcript_against_captions(
        transcripts,
        "This session is resumed.",
        min_similarity=45.0,
        max_seconds=None,
    )

    assert result.status == "empty_transcript"


def test_guardrail_respects_max_seconds() -> None:
    transcripts = [
        Transcript(start="0:00:05", text="This session is resumed.", voice=1),
        Transcript(start="0:15:00", text="Unrelated later text.", voice=1),
    ]

    result = validate_transcript_against_captions(
        transcripts,
        "This session is resumed.",
        min_similarity=45.0,
        max_seconds=600,
    )

    assert result.status == "ok"
