#!/usr/bin/env python3
#!/usr/bin/env python3
# ruff: noqa: E402

"""Backfill session-scoped speaker roles for an existing video.

This lets you attach order-paper-derived roles to a `youtube_video_id` without
reprocessing a full transcript or regenerating the knowledge graph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from rapidfuzz import fuzz


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.postgres_client import PostgresClient
from lib.order_papers.role_extract import extract_speaker_roles
from lib.roles import (
    infer_role_kind,
    normalize_person_name,
    normalize_role_label,
    split_role_labels,
)


def _load_order_paper_parsed_json(
    postgres: PostgresClient, order_paper_id: str
) -> dict:
    rows = postgres.execute_query(
        """
        SELECT parsed_json
        FROM order_papers
        WHERE id = %s
        """,
        (order_paper_id,),
    )
    if not rows:
        raise ValueError(f"Order paper not found: {order_paper_id}")

    parsed_json = rows[0][0]
    if parsed_json is None:
        return {}
    if isinstance(parsed_json, dict):
        return parsed_json
    if isinstance(parsed_json, str):
        try:
            return json.loads(parsed_json)
        except Exception:
            return {}
    return {}


def _build_exact_name_to_speaker_id_map(
    postgres: PostgresClient, *, youtube_video_id: str
) -> dict[str, str]:
    rows = postgres.execute_query(
        """
        SELECT DISTINCT
            s.speaker_id,
            COALESCE(sp.full_name, sp.normalized_name, s.speaker_id) AS speaker_name
        FROM sentences s
        LEFT JOIN speakers sp ON sp.id = s.speaker_id
        WHERE s.youtube_video_id = %s
        """,
        (youtube_video_id,),
    )
    out: dict[str, str] = {}
    for speaker_id, name in rows:
        norm = normalize_person_name(str(name or "").strip())
        if norm:
            out[norm] = str(speaker_id)
    return out


def _strip_honorifics_and_punct(name: str) -> str:
    norm = normalize_person_name(name)
    for ch in [".", "'", "-", "_"]:
        norm = norm.replace(ch, " ")
    norm = " ".join(norm.split())

    honorifics = {
        "hon",
        "the",
        "rt",
        "right",
        "dr",
        "sir",
        "mr",
        "mrs",
        "ms",
        "his",
        "her",
        "honour",
    }
    tokens = [t for t in norm.split() if t not in honorifics]
    return " ".join(tokens)


def _surname(tokenized_name: str) -> str:
    toks = [t for t in tokenized_name.split() if t]
    return toks[-1] if toks else ""


def _match_speaker_id(
    *,
    speaker_name: str,
    name_to_id_exact: dict[str, str],
    candidates: list[tuple[str, str]],
) -> str | None:
    exact = name_to_id_exact.get(normalize_person_name(speaker_name))
    if exact:
        return exact

    cleaned = _strip_honorifics_and_punct(speaker_name)
    cleaned_surname = _surname(cleaned)
    cleaned_tokens = cleaned.split()
    first_initial = cleaned_tokens[0][:1] if cleaned_tokens else ""

    if cleaned_surname:
        surname_matches: list[str] = []
        for speaker_id, candidate_name in candidates:
            cand_clean = _strip_honorifics_and_punct(candidate_name)
            if _surname(cand_clean) != cleaned_surname:
                continue
            if not first_initial:
                surname_matches.append(speaker_id)
                continue
            cand_tokens = cand_clean.split()
            cand_initial = cand_tokens[0][:1] if cand_tokens else ""
            if cand_initial == first_initial:
                surname_matches.append(speaker_id)

        if len(surname_matches) == 1:
            return surname_matches[0]

    best_id: str | None = None
    best_score = 0
    for speaker_id, candidate_name in candidates:
        score = fuzz.token_sort_ratio(
            cleaned, _strip_honorifics_and_punct(candidate_name)
        )
        if score > best_score:
            best_score = score
            best_id = speaker_id

    if best_id and best_score >= 88:
        return best_id
    return None


def backfill_roles_for_video(
    *,
    youtube_video_id: str,
    order_paper_id: str,
    dry_run: bool,
) -> int:
    with PostgresClient() as pg:
        parsed_json = _load_order_paper_parsed_json(pg, order_paper_id)
        roles = extract_speaker_roles(parsed_json)
        if not roles:
            print("⚠️ No speaker roles found in order paper parsed_json")
            return 0

        name_to_id = _build_exact_name_to_speaker_id_map(
            pg, youtube_video_id=youtube_video_id
        )
        candidates = [(sid, nm) for nm, sid in name_to_id.items()]

        inserted = 0
        for speaker_name, role_label_raw in roles:
            speaker_name_norm = normalize_person_name(speaker_name)
            if not speaker_name_norm:
                continue

            speaker_id = _match_speaker_id(
                speaker_name=speaker_name,
                name_to_id_exact=name_to_id,
                candidates=candidates,
            )

            role_labels = split_role_labels(role_label_raw)
            if not role_labels:
                continue

            for role_label in role_labels:
                role_label_norm = normalize_role_label(role_label)
                if not role_label_norm:
                    continue

                if role_label_norm in {"sc", "s.c.", "kc", "k.c.", "jp", "j.p."}:
                    continue

                role_kind = infer_role_kind(role_label)

                if dry_run:
                    print(
                        f"DRY RUN: {youtube_video_id} | {speaker_name} | {role_label} "
                        f"({role_kind}) -> {speaker_id or 'NULL'}"
                    )
                    inserted += 1
                    continue

                pg.execute_update(
                    """
                    INSERT INTO speaker_video_roles (
                        youtube_video_id,
                        speaker_id,
                        speaker_name_raw,
                        speaker_name_norm,
                        role_label,
                        role_label_norm,
                        role_kind,
                        source,
                        source_id,
                        confidence,
                        evidence,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, NOW())
                    ON CONFLICT (
                        youtube_video_id,
                        speaker_name_norm,
                        role_kind,
                        role_label_norm,
                        source,
                        source_id
                    )
                    DO UPDATE SET
                        speaker_id = COALESCE(EXCLUDED.speaker_id, speaker_video_roles.speaker_id),
                        speaker_name_raw = EXCLUDED.speaker_name_raw,
                        role_label = EXCLUDED.role_label,
                        updated_at = NOW()
                    """,
                    (
                        youtube_video_id,
                        speaker_id,
                        speaker_name,
                        speaker_name_norm,
                        role_label,
                        role_label_norm,
                        role_kind,
                        "order_paper",
                        order_paper_id,
                    ),
                )
                inserted += 1

        return inserted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill speaker_video_roles for an existing video from an existing order paper"
    )
    parser.add_argument("--youtube-video-id", required=True, help="YouTube video ID")
    parser.add_argument(
        "--order-paper-id", required=True, help="Order paper ID (order_papers.id)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be inserted"
    )
    args = parser.parse_args()

    try:
        count = backfill_roles_for_video(
            youtube_video_id=args.youtube_video_id,
            order_paper_id=args.order_paper_id,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(f"✅ Dry-run complete; would upsert {count} roles")
        else:
            print(f"✅ Upserted {count} roles into speaker_video_roles")
        return 0
    except Exception as e:
        print(f"❌ Backfill failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
