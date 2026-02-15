"""Clean transcript JSON timestamps and emit corrected files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.transcripts.timestamp_fix import (  # noqa: E402
    build_sentence_id_mapping,
    parse_duration_to_seconds,
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _clean_transcript_file(path: Path, output_dir: Path) -> dict:
    data = _load_json(path)
    transcripts = data.get("transcripts", [])
    video_id = path.stem.replace("transcription_output_", "")
    duration = data.get("video_metadata", {}).get("duration", "")
    duration_seconds = parse_duration_to_seconds(duration)

    mapping = build_sentence_id_mapping(
        transcripts=transcripts,
        youtube_video_id=video_id,
        duration_seconds=duration_seconds,
    )

    cleaned = dict(data)
    cleaned["transcripts"] = mapping.cleaned_transcripts

    out_path = output_dir / path.name
    _write_json(out_path, cleaned)

    corrected = sum(1 for reason in mapping.reasons if reason != "hms" and reason != "mmss")
    return {
        "file": str(path),
        "output": str(out_path),
        "total": len(mapping.reasons),
        "corrected": corrected,
        "duration_seconds": duration_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean transcript timestamps and emit JSON")
    parser.add_argument(
        "--input",
        help="Single transcript JSON file to clean",
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory to scan for transcription_output_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        default="cleaned_transcripts",
        help="Directory for cleaned JSON files",
    )
    parser.add_argument(
        "--report-file",
        default="cleaned_transcripts/clean_report.json",
        help="Path to write a summary report",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reports = []

    if args.input:
        reports.append(_clean_transcript_file(Path(args.input), output_dir))
    else:
        input_dir = Path(args.input_dir)
        for path in sorted(input_dir.glob("transcription_output_*.json")):
            reports.append(_clean_transcript_file(path, output_dir))

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"files": reports}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"✅ Cleaned {len(reports)} file(s)")
    print(f"   Output dir: {output_dir}")
    print(f"   Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
