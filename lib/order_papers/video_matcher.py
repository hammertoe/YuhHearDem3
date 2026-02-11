"""Automatic order-paper matching for a video."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
from typing import Any

from lib.db.postgres_client import PostgresClient


class MatchConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MatchStatus(str, Enum):
    AUTO_MATCHED = "auto_matched"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class MatchCandidate:
    order_paper_id: str
    score: float
    reasons: list[str]


@dataclass(frozen=True)
class MatchDecision:
    youtube_video_id: str
    order_paper_id: str | None
    score: float
    confidence: MatchConfidence
    status: MatchStatus
    reasons: list[str]
    candidates: list[MatchCandidate]


def _score_upload_date(
    upload_date: date | None,
    sitting_date: date | None,
) -> tuple[float, str]:
    if not upload_date or not sitting_date:
        return 0.0, "no upload-date signal"

    day_delta = abs((upload_date - sitting_date).days)
    if day_delta == 0:
        return 55.0, "exact upload-date match"
    if day_delta == 1:
        return 35.0, "1-day upload-date delta"
    if day_delta <= 3:
        return 20.0, "<=3-day upload-date delta"
    if day_delta <= 7:
        return 8.0, "<=7-day upload-date delta"
    return 0.0, ">7-day upload-date delta"


_TITLE_DATE_RE = re.compile(
    r"\b(?:mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?)\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"[,]?\s+(\d{4})\b",
    re.IGNORECASE,
)


_MONTH_TO_INT = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _extract_title_date(video_title: str) -> date | None:
    title = (video_title or "").strip()
    if not title:
        return None

    match = _TITLE_DATE_RE.search(title)
    if not match:
        return None

    day_str, month_str, year_str = match.groups()
    try:
        return date(int(year_str), _MONTH_TO_INT[month_str.lower()], int(day_str))
    except Exception:
        return None


def _score_title(
    video_title: str,
    session_text: str,
    sitting_date: date | None,
) -> tuple[float, str]:
    title = (video_title or "").lower()
    session = (session_text or "").lower()

    score = 0.0
    reasons: list[str] = []

    title_date = _extract_title_date(video_title)
    if title_date and sitting_date:
        day_delta = abs((title_date - sitting_date).days)
        if day_delta == 0:
            score += 35.0
            reasons.append("exact title-date match")
        elif day_delta == 1:
            score += 20.0
            reasons.append("1-day title-date delta")
        elif day_delta <= 3:
            score += 8.0
            reasons.append("<=3-day title-date delta")
        else:
            reasons.append(">3-day title-date delta")
    else:
        reasons.append("no title-date signal")

    chamber_words = ["house", "senate", "assembly"]
    shared_chamber = bool(title and session) and any(
        w in title and w in session for w in chamber_words
    )
    if shared_chamber:
        score += 10.0
        reasons.append("chamber words align")
    else:
        reasons.append("no chamber alignment")

    return score, "; ".join(reasons)


def _to_confidence(score: float) -> MatchConfidence:
    if score >= 75:
        return MatchConfidence.HIGH
    if score >= 60:
        return MatchConfidence.MEDIUM
    return MatchConfidence.LOW


def _to_status(confidence: MatchConfidence) -> MatchStatus:
    if confidence == MatchConfidence.HIGH:
        return MatchStatus.AUTO_MATCHED
    return MatchStatus.NEEDS_REVIEW


def _load_video_meta(
    postgres: PostgresClient,
    *,
    youtube_video_id: str,
) -> tuple[str, date | None]:
    rows = postgres.execute_query(
        """
        SELECT title, upload_date
        FROM videos
        WHERE youtube_id = %s
        """,
        (youtube_video_id,),
    )
    if not rows:
        raise ValueError(f"Video not found in videos table: {youtube_video_id}")
    title, upload_date = rows[0]
    return str(title or ""), upload_date


def _coalesce_base_date(*, upload_date: date | None, video_title: str) -> date | None:
    if upload_date:
        return upload_date
    return _extract_title_date(video_title)


def _load_candidate_order_papers(
    postgres: PostgresClient,
    *,
    video_date: date | None,
    limit: int,
) -> list[tuple[str, date | None, str, Any]]:
    if video_date:
        return postgres.execute_query(
            """
            SELECT id, sitting_date, session, parsed_json
            FROM order_papers
            WHERE sitting_date BETWEEN (%s::date - INTERVAL '7 days') AND (%s::date + INTERVAL '7 days')
            ORDER BY ABS((sitting_date - %s::date)), updated_at DESC
            LIMIT %s
            """,
            (video_date, video_date, video_date, int(limit)),
        )

    return postgres.execute_query(
        """
        SELECT id, sitting_date, session, parsed_json
        FROM order_papers
        ORDER BY updated_at DESC
        LIMIT %s
        """,
        (int(limit),),
    )


def _persist_match_decision(postgres: PostgresClient, decision: MatchDecision) -> None:
    postgres.execute_update(
        """
        INSERT INTO video_order_paper_matches (
            youtube_video_id,
            order_paper_id,
            score,
            confidence,
            status,
            reasons,
            candidate_scores,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW())
        ON CONFLICT (youtube_video_id)
        DO UPDATE SET
            order_paper_id = EXCLUDED.order_paper_id,
            score = EXCLUDED.score,
            confidence = EXCLUDED.confidence,
            status = EXCLUDED.status,
            reasons = EXCLUDED.reasons,
            candidate_scores = EXCLUDED.candidate_scores,
            updated_at = NOW()
        WHERE video_order_paper_matches.status <> 'manual_override'
        """,
        (
            decision.youtube_video_id,
            decision.order_paper_id,
            decision.score,
            decision.confidence.value,
            decision.status.value,
            json.dumps(decision.reasons),
            json.dumps([asdict(c) for c in decision.candidates]),
        ),
    )


def match_order_paper_for_video(
    postgres: PostgresClient,
    *,
    youtube_video_id: str,
    persist: bool = True,
    max_candidates: int = 25,
) -> MatchDecision:
    """Return the best order-paper match for a video."""

    video_title, video_date = _load_video_meta(
        postgres, youtube_video_id=youtube_video_id
    )
    return match_order_paper_for_video_metadata(
        postgres,
        youtube_video_id=youtube_video_id,
        video_title=video_title,
        upload_date=video_date,
        persist=persist,
        max_candidates=max_candidates,
    )


def match_order_paper_for_video_metadata(
    postgres: PostgresClient,
    *,
    youtube_video_id: str,
    video_title: str,
    upload_date: date | None = None,
    persist: bool = True,
    max_candidates: int = 25,
) -> MatchDecision:
    """Return the best order-paper match from provided video metadata."""

    base_date = _coalesce_base_date(upload_date=upload_date, video_title=video_title)
    candidates = _load_candidate_order_papers(
        postgres,
        video_date=base_date,
        limit=max_candidates,
    )

    ranked: list[MatchCandidate] = []
    for order_paper_id, sitting_date, session_text, _parsed_json in candidates:
        upload_date_score, upload_date_reason = _score_upload_date(
            base_date, sitting_date
        )
        title_score, title_reason = _score_title(
            video_title,
            str(session_text or ""),
            sitting_date,
        )

        score = min(100.0, upload_date_score + title_score)
        ranked.append(
            MatchCandidate(
                order_paper_id=str(order_paper_id),
                score=score,
                reasons=[upload_date_reason, title_reason],
            )
        )

    ranked.sort(key=lambda c: c.score, reverse=True)

    if not ranked:
        decision = MatchDecision(
            youtube_video_id=youtube_video_id,
            order_paper_id=None,
            score=0.0,
            confidence=MatchConfidence.LOW,
            status=MatchStatus.NEEDS_REVIEW,
            reasons=["no candidate order papers found"],
            candidates=[],
        )
        if persist:
            _persist_match_decision(postgres, decision)
        return decision

    top = ranked[0]
    confidence = _to_confidence(top.score)
    status = _to_status(confidence)

    decision = MatchDecision(
        youtube_video_id=youtube_video_id,
        order_paper_id=top.order_paper_id,
        score=top.score,
        confidence=confidence,
        status=status,
        reasons=top.reasons,
        candidates=ranked[:5],
    )

    if persist:
        _persist_match_decision(postgres, decision)

    return decision
