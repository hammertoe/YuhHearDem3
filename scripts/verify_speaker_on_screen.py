"""Verify speakers using on-screen text from short video clips."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai.types import FileData, GenerateContentConfig, Part, ThinkingConfig, VideoMetadata

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.id_generators import format_seconds_to_timestamp  # noqa: E402
from lib.transcripts.speaker_verification import (  # noqa: E402
    SpeakerVerificationResult,
    build_verification_prompt,
    is_actionable_result,
    parse_flags_csv,
)


YOUTUBE_URL_PREFIX = "https://www.youtube.com/watch?v="


@dataclass(frozen=True)
class VerificationRecord:
    video_id: str
    speaker_id: str
    timestamp_str: str
    requested_name: str
    result: SpeakerVerificationResult | None
    error: str | None


def _youtube_url(video_id: str) -> str:
    return f"{YOUTUBE_URL_PREFIX}{video_id}"


def _get_video_part(video_id: str, start_seconds: int, duration_seconds: int = 1) -> Part:
    file_data = FileData(file_uri=_youtube_url(video_id), mime_type="video/*")
    start_offset = f"{start_seconds}s"
    end_offset = f"{start_seconds + duration_seconds}s"
    return Part(
        file_data=file_data,
        video_metadata=VideoMetadata(start_offset=start_offset, end_offset=end_offset),
    )


def _load_video_titles(video_ids: list[str]) -> dict[str, str]:
    if not video_ids:
        return {}
    from lib.db.postgres_client import PostgresClient

    with PostgresClient() as pg:
        rows = pg.execute_query(
            "SELECT youtube_id, title FROM videos WHERE youtube_id = ANY(%s)",
            (video_ids,),
        )
    return {row[0]: row[1] or "" for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify speakers using on-screen text")
    parser.add_argument("--flags-csv", required=True, help="CSV from flag_speaker_mismatches")
    parser.add_argument("--output", default="speaker_verification_results.json")
    parser.add_argument("--video-id", help="Comma-separated list of video IDs")
    parser.add_argument("--max-per-speaker", type=int, default=1)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--actionable-only", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.85)
    args = parser.parse_args()

    flags = parse_flags_csv(Path(args.flags_csv))
    filter_ids = {v.strip() for v in (args.video_id or "").split(",") if v.strip()}
    if filter_ids:
        flags = [flag for flag in flags if flag.video_id in filter_ids]

    if not flags:
        print("⚠️  No flags to verify")
        return 0

    video_titles = _load_video_titles(sorted({flag.video_id for flag in flags}))
    client = genai.Client()
    config = GenerateContentConfig(
        temperature=0.0,
        top_p=0.0,
        response_mime_type="application/json",
        response_schema=SpeakerVerificationResult,
        thinking_config=ThinkingConfig(thinking_budget=0, include_thoughts=False),
    )

    output: list[dict[str, object]] = []
    processed = 0
    total_limit = args.max_total if args.max_total and args.max_total > 0 else None

    try:
        for flag in flags:
            samples = flag.sample_seconds[: args.max_per_speaker]
            for seconds in samples:
                if total_limit is not None and processed >= total_limit:
                    raise StopIteration
                processed += 1
                timestamp_str = format_seconds_to_timestamp(seconds)
                prompt = build_verification_prompt(
                    video_titles.get(flag.video_id, ""),
                    timestamp_str,
                    flag.speaker_name,
                )
                part = _get_video_part(flag.video_id, seconds, duration_seconds=1)

                print(
                    f"→ {flag.video_id} {timestamp_str} ({flag.speaker_id} / {flag.speaker_name})"
                )

                record = VerificationRecord(
                    video_id=flag.video_id,
                    speaker_id=flag.speaker_id,
                    timestamp_str=timestamp_str,
                    requested_name=flag.speaker_name,
                    result=None,
                    error=None,
                )

                try:
                    response = client.models.generate_content(
                        model=args.model,
                        contents=[part, prompt],
                        config=config,
                    )
                    parsed = response.parsed if hasattr(response, "parsed") else None
                    if isinstance(parsed, SpeakerVerificationResult):
                        record = record.__class__(
                            **{**record.__dict__, "result": parsed, "error": None}
                        )
                    else:
                        record = record.__class__(
                            **{**record.__dict__, "error": "No parsed response"}
                        )
                except Exception as exc:
                    record = record.__class__(**{**record.__dict__, "error": str(exc)})

                actionable = is_actionable_result(record.result, args.min_confidence)
                if args.actionable_only and not actionable:
                    continue

                output.append(
                    {
                        "video_id": record.video_id,
                        "speaker_id": record.speaker_id,
                        "timestamp": record.timestamp_str,
                        "requested_name": record.requested_name,
                        "result": record.result.model_dump() if record.result else None,
                        "error": record.error,
                    }
                )

                payload = {
                    "generated_at": datetime.now().isoformat(),
                    "model": args.model,
                    "actionable_only": args.actionable_only,
                    "min_confidence": args.min_confidence,
                    "results": output,
                }
                Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"   saved → {args.output}")
    except StopIteration:
        pass

    if output:
        print(f"✅ Wrote {len(output)} results to {args.output}")
    else:
        print("⚠️  No results generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
