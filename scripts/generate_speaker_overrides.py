"""Generate override JSON from speaker verification results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.transcripts.override_generation import (  # noqa: E402
    build_override_entries,
    load_verification_results,
    map_overrides_to_speaker_ids,
)
from lib.id_generators import normalize_label  # noqa: E402
from lib.db.postgres_client import PostgresClient  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate speaker overrides JSON")
    parser.add_argument("--verification", required=True, help="verification results JSON")
    parser.add_argument("--output", default="speaker_overrides_generated.json")
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--allow-fuzzy", action="store_true")
    parser.add_argument("--min-name-similarity", type=float, default=92.0)
    args = parser.parse_args()

    results = load_verification_results(Path(args.verification))
    overrides = build_override_entries(results, args.min_confidence)

    with PostgresClient() as pg:
        rows = pg.execute_query("SELECT id, full_name FROM speakers")
    name_to_id: dict[str, str] = {}
    normalized_to_id: dict[str, str] = {}
    for speaker_id, full_name in rows:
        if not full_name:
            continue
        full_name_str = str(full_name).strip()
        name_to_id[full_name_str] = speaker_id
        normalized_to_id[normalize_label(full_name_str)] = speaker_id

    resolved, unresolved = map_overrides_to_speaker_ids(overrides, name_to_id, normalized_to_id)

    if unresolved and args.allow_fuzzy:
        normalized_names = list(normalized_to_id.keys())
        still_unresolved: list[dict[str, object]] = []
        for entry in unresolved:
            raw_name = str(entry.get("new_speaker_name") or "").strip()
            norm_name = normalize_label(raw_name)
            match = process.extractOne(
                norm_name,
                normalized_names,
                scorer=fuzz.ratio,
            )
            if match is None:
                still_unresolved.append(entry)
                continue
            candidate, score, _ = match
            if score >= args.min_name_similarity:
                resolved.append(
                    {
                        **entry,
                        "new_speaker_id": normalized_to_id[candidate],
                        "note": f"fuzzy_match:{candidate}:{score:.1f}",
                    }
                )
            else:
                still_unresolved.append(
                    {
                        **entry,
                        "suggested_name": candidate,
                        "suggested_score": score,
                    }
                )
        unresolved = still_unresolved

    payload = {
        "generated_at": datetime.now().isoformat(),
        "min_confidence": args.min_confidence,
        "overrides": resolved,
        "unresolved": unresolved,
        "allow_fuzzy": args.allow_fuzzy,
        "min_name_similarity": args.min_name_similarity,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(resolved)} overrides to {args.output}")
    if unresolved:
        print(f"⚠️  {len(unresolved)} overrides unresolved (missing speaker id)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
