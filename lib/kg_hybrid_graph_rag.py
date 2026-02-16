from __future__ import annotations

import re
import json
from typing import Any

from lib.db.pgvector import vector_literal
from lib.id_generators import normalize_label

# Topic terms for boost detection (Barbados-specific)
TOPIC_TERMS: set[str] = {
    "water",
    "bwa",
    "drought",
    "irrigation",
    "flood",
    "drainage",
    "housing",
    "home",
    "homeless",
    "property",
    "rent",
    "squatter",
    "tourism",
    "visitor",
    "hotel",
    "cruise",
    "beach",
    "coast",
    "education",
    "school",
    "university",
    "student",
    "teacher",
    "health",
    "hospital",
    "doctor",
    "nurse",
    "clinic",
    "cancer",
    "hiv",
    "economy",
    "gdp",
    "inflation",
    "tax",
    "revenue",
    "budget",
    "crime",
    "police",
    "prison",
    "murder",
    "robbery",
    "gang",
    "climate",
    "climate-change",
    "carbon",
    "emission",
    "renewable",
    "solar",
    "agriculture",
    "farm",
    "crop",
    "livestock",
    "food",
    "fishing",
    "transport",
    "road",
    "bus",
    "traffic",
    "bridge",
    "airport",
    "employment",
    "job",
    "unemployment",
    "worker",
    "labour",
    "social",
    "welfare",
    "pension",
    "elderly",
    "disabled",
}

# Generic governance terms that should be penalized when no topic overlap
GENERIC_GOVERNANCE_TERMS: set[str] = {
    "minister",
    "ministers",
    "ministry",
    "government",
    "cabinet",
    "parliament",
    "house",
    "senate",
    "bill",
    "legislation",
    "law",
    "regulation",
    "policy",
    "amendment",
    "motion",
    "debate",
    "speech",
    "member",
    "mp",
    "senator",
    "speaker",
    "chairman",
    "chairperson",
    "resolution",
    "committee",
    "report",
    "order",
    "paper",
}


def _extract_query_intent(query: str) -> dict[str, Any]:
    """Extract topic terms and recency intent from query."""
    query_lower = (query or "").lower()
    terms = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b", query_lower))

    topic_matches = terms & TOPIC_TERMS
    generic_matches = terms & GENERIC_GOVERNANCE_TERMS

    has_recency = any(
        t in query_lower for t in {"recent", "recently", "latest", "last", "new", "current"}
    )

    return {
        "topic_terms": list(topic_matches),
        "generic_terms": list(generic_matches),
        "has_recency": has_recency,
        "is_topical": len(topic_matches) > 0,
    }


def _fuse_candidates_rrf(
    vector_candidates: list[dict[str, Any]],
    fulltext_candidates: list[dict[str, Any]],
    alias_candidates: list[dict[str, Any]],
    query: str,
    k: int = 60,
    intent: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked candidate lists using Reciprocal Rank Fusion with boosts."""
    intent = intent or {}
    topic_terms = set(intent.get("topic_terms", []))
    has_recency = intent.get("has_recency", False)
    is_topical = intent.get("is_topical", False)
    generic_terms = set(intent.get("generic_terms", []))

    def get_rrf_scores(items: list[dict[str, Any]], channel: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        for rank, item in enumerate(items, 1):
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            rrf = 1.0 / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf
        return scores

    vec_scores = get_rrf_scores(vector_candidates, "vector")
    ft_scores = get_rrf_scores(fulltext_candidates, "fulltext")
    alias_scores = get_rrf_scores(alias_candidates, "alias")

    all_ids: set[str] = set(vec_scores) | set(ft_scores) | set(alias_scores)

    def build_item_map(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(item.get("id", "")): item for item in items if item.get("id")}

    vec_map = build_item_map(vector_candidates)
    ft_map = build_item_map(fulltext_candidates)
    alias_map = build_item_map(alias_candidates)

    fused: list[dict[str, Any]] = []
    for item_id in all_ids:
        rrf_score = (
            vec_scores.get(item_id, 0.0)
            + ft_scores.get(item_id, 0.0)
            + alias_scores.get(item_id, 0.0)
        )

        base_item = vec_map.get(item_id) or ft_map.get(item_id) or alias_map.get(item_id)
        if not base_item:
            continue

        label = (base_item.get("label") or "").lower()
        aliases = base_item.get("aliases") or []
        aliases_text = " ".join(aliases).lower() if aliases else ""
        item_text = f"{label} {aliases_text}"

        boost = 0.0

        if topic_terms:
            item_terms = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b", item_text))
            topic_coverage = len(topic_terms & item_terms)
            if topic_coverage > 0:
                boost += 0.10 * min(topic_coverage, 3)

            if any(t in item_text for t in topic_terms):
                boost += 0.05

        if has_recency:
            boost += 0.03

        if generic_terms and is_topical and not topic_terms:
            item_terms = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b", item_text))
            if not (topic_terms & item_terms):
                boost -= 0.05

        final_score = rrf_score + boost
        fused.append(
            {**base_item, "fused_score": final_score, "rrf_score": rrf_score, "boost": boost}
        )

    fused.sort(key=lambda x: float(x.get("fused_score", 0.0)), reverse=True)
    return fused


def _rerank_with_gemini(
    candidates: list[dict[str, Any]],
    query: str,
    model: str = "gemini-2.0-flash",
    top_n: int = 40,
    timeout_ms: int = 3000,
) -> list[dict[str, Any]]:
    """Rerank top candidates using Gemini with strict JSON scoring."""
    if not candidates or len(candidates) < 3:
        return candidates[:top_n]

    top_candidates = candidates[: min(len(candidates), top_n)]
    intent = _extract_query_intent(query)

    candidate_list = "\n".join(
        f"- {i + 1}. [{c.get('id')}] {c.get('label')} (type: {c.get('type')})"
        for i, c in enumerate(top_candidates)
    )

    prompt = f"""You are a relevance scorer for a parliamentary knowledge graph.
Query: "{query}"

Topic terms in query: {intent.get("topic_terms", [])}
Has recency intent: {intent.get("has_recency", False)}

Candidates to score (respond with ONLY valid JSON array, no other text):
{candidate_list}

For each candidate, score from 0-1 on:
- relevance: How directly relevant to the query topic
- topic_match: Does it match the topic terms in the query?
- specificity: Is it specific (not generic governance)?
- recency_fit: Does it fit recent discussions? (if recency intent)

Respond with ONLY a JSON array of objects with fields: id, relevance, topic_match, specificity, recency_fit, reason (short phrase).
Example: [{{"id": "kg_123", "relevance": 0.9, "topic_match": 0.8, "specificity": 0.7, "recency_fit": 0.5, "reason": "direct water topic"}}]
"""

    try:
        from google import genai
        from lib.utils.config import config

        client = genai.Client(api_key=config.embedding.api_key)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
                "max_output_tokens": 4000,
            },
        )

        text = ""
        if hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = getattr(candidate.content, "parts", None)
                if parts:
                    for part in parts:
                        if hasattr(part, "text") and part.text:
                            text += part.text

        if not text:
            return candidates[:top_n]

        scores = json.loads(text)
        if not isinstance(scores, list):
            return candidates[:top_n]

        score_map = {str(s.get("id", "")): s for s in scores if s.get("id")}

        reranked: list[dict[str, Any]] = []
        for c in top_candidates:
            c_id = str(c.get("id", ""))
            s = score_map.get(c_id, {})
            relevance = float(s.get("relevance", 0.5))
            topic_match = float(s.get("topic_match", 0.5))
            specificity = float(s.get("specificity", 0.5))
            recency_fit = float(s.get("recency_fit", 0.5))

            final_score = (
                0.55 * relevance + 0.20 * topic_match + 0.15 * specificity + 0.10 * recency_fit
            )
            reranked.append({**c, "rerank_score": final_score, "rerank_details": s})

        reranked.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        return reranked

    except Exception:
        return candidates[:top_n]


def build_youtube_url(youtube_video_id: str, seconds_since_start: int) -> str:
    seconds = int(seconds_since_start)
    if seconds < 0:
        seconds = 0
    return f"https://www.youtube.com/watch?v={youtube_video_id}&t={seconds}s"


def format_speaker_name(
    *,
    full_name: str | None,
    normalized_name: str | None,
    speaker_id: str | None,
) -> str:
    full = (full_name or "").strip()
    if full:
        return full

    raw = (normalized_name or "").strip()
    if not raw:
        return (speaker_id or "").strip()

    # If the "normalized" value is actually a speaker_id, convert it.
    if raw.startswith("s_"):
        raw = _speaker_id_to_name_guess(raw)

    words = raw.split()
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in {"hon", "honourable", "hon."}:
            out.append("The")
            out.append("Honourable")
        elif lw in {"mr", "mister"}:
            out.append("Mr.")
        elif lw in {"ms", "miss"}:
            out.append("Ms.")
        elif lw == "mrs":
            out.append("Mrs.")
        elif lw == "dr":
            out.append("Dr.")
        else:
            out.append(w[:1].upper() + w[1:])

    # Collapse accidental duplicated leading tokens (e.g. "The Honourable The House")
    return " ".join(out).strip()


def _speaker_id_to_name_guess(speaker_id: str | None) -> str:
    sid = (speaker_id or "").strip()
    if not sid:
        return ""
    sid = re.sub(r"^s_", "", sid)
    sid = re.sub(r"_\d+$", "", sid)
    return sid.replace("_", " ").strip()


def _strip_honorific_prefix(name_norm: str) -> str:
    s = (name_norm or "").strip()
    s = re.sub(
        r"^(the\s+)?(most\s+)?(honourable|hon\.?|mr\.?|ms\.?|mrs\.?|dr\.?|senator)\s+",
        "",
        s,
    )
    return s.strip()


def _smart_titlecase_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""

    def fix_token(tok: str) -> str:
        if not tok:
            return tok
        # Preserve initials and abbreviations like "L." or "R." or "St.".
        if "." in tok and len(tok) <= 4:
            return tok.upper()
        if tok.isupper() and len(tok) <= 3:
            return tok
        # Handle hyphenated and apostrophe names.
        parts = re.split(r"([-'])", tok)
        out: list[str] = []
        for p in parts:
            if p in {"-", "'"}:
                out.append(p)
            elif p:
                out.append(p[:1].upper() + p[1:].lower())
        return "".join(out)

    tokens = re.split(r"\s+", raw)
    return " ".join(fix_token(t) for t in tokens if t)


def _format_title_and_name(title: str | None, name: str) -> str:
    nm = _smart_titlecase_name(name)
    t = (title or "").strip()
    if not t:
        return nm

    tl = t.lower()
    if "most" in tl and "hon" in tl:
        return f"The Most Honourable {nm}".strip()
    if tl.startswith("hon") or "hon" in tl or "honourable" in tl:
        return f"The Honourable {nm}".strip()
    if tl.startswith("mr"):
        return f"Mr. {nm}".strip()
    if tl.startswith("ms"):
        return f"Ms. {nm}".strip()
    if tl.startswith("mrs"):
        return f"Mrs. {nm}".strip()
    if tl.startswith("dr"):
        return f"Dr. {nm}".strip()
    return f"{_smart_titlecase_name(t)} {nm}".strip()


def _load_order_paper_speaker_index(
    *,
    postgres: Any,
    limit_order_papers: int = 25,
) -> dict[str, str]:
    """Best-effort index of speakers from recent order papers.

    Returns a mapping of normalized keys -> display name.
    """

    try:
        rows = postgres.execute_query(
            """
            WITH recent_ops AS (
                SELECT sitting_date, parsed_json
                FROM order_papers
                WHERE parsed_json ? 'speakers'
                ORDER BY sitting_date DESC
                LIMIT %s
            )
            SELECT recent_ops.sitting_date::text,
                   sp->>'name' AS name,
                   sp->>'title' AS title,
                   sp->>'role' AS role
            FROM recent_ops
            CROSS JOIN LATERAL jsonb_array_elements(recent_ops.parsed_json->'speakers') sp
            """,
            (int(limit_order_papers),),
        )
    except Exception:
        return {}

    idx: dict[str, str] = {}

    def add_key(key: str, display: str) -> None:
        k = normalize_label(key)
        if not k:
            return
        # Prefer first-seen (which is most recent due to ORDER BY sitting_date DESC).
        idx.setdefault(k, display)

    for _date_str, name, title, _role in rows:
        nm = (name or "").strip()
        if not nm:
            continue
        display = _format_title_and_name(title, nm)

        add_key(nm, display)
        if title:
            add_key(f"{title} {nm}", display)

            # Common shorthand tokens to match transcript speaker IDs.
            tl = str(title).lower()
            if tl.startswith("hon") or "honourable" in tl:
                add_key(f"hon {nm}", display)

    # Also add stripped-honorific variants to improve matching.
    for k, v in list(idx.items()):
        stripped = _strip_honorific_prefix(k)
        if stripped and stripped != k:
            idx.setdefault(stripped, v)

    return idx


def _dedupe_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_scores: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get("id", ""))
        if not item_id:
            continue
        item_score = float(item.get("score", 0.0))
        if item_id not in best_scores or item_score > float(best_scores[item_id].get("score", 0.0)):
            best_scores[item_id] = item
    return list(best_scores.values())


def _query_terms(query: str) -> list[str]:
    terms = [t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b", query or "")]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "about",
        "into",
        "over",
        "under",
        "have",
        "has",
        "had",
    }
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in stop or term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def _url_with_page_fragment(url: str, page_number: int | None) -> str:
    base = str(url or "").strip()
    page = int(page_number or 0)
    if not base or page <= 0:
        return base
    if re.search(r"#page=\d+", base):
        return base
    return f"{base}#page={page}"


def _retrieve_seed_nodes(
    *,
    postgres: Any,
    embedding_client: Any,
    query: str,
    seed_k: int,
    enable_rerank: bool = True,
    rerank_model: str = "gemini-2.0-flash",
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    vector_candidates: list[dict[str, Any]] = []
    fulltext_candidates: list[dict[str, Any]] = []
    alias_candidates: list[dict[str, Any]] = []

    # Vector search (best when embeddings exist)
    try:
        embedding = query_embedding
        if embedding is None:
            embedding = embedding_client.generate_query_embedding(query)
        rows = postgres.execute_query(
            """
            SELECT id, type, label, aliases, embedding <=> %s AS distance
            FROM kg_nodes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s ASC
            LIMIT %s
            """,
            (vector_literal(embedding), vector_literal(embedding), seed_k * 2),
        )
        for row in rows:
            distance = float(row[4] or 0.0)
            vector_candidates.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "label": row[2],
                    "aliases": row[3] or [],
                    "score": 1.0 - distance,
                    "match_reason": "vector",
                }
            )
    except Exception:
        pass

    # Full-text search on label+aliases (kg_nodes.tsv trigger)
    rows = postgres.execute_query(
        """
        SELECT id, type, label, aliases, ts_rank(tsv, plainto_tsquery('english', %s)) as rank
        FROM kg_nodes
        WHERE tsv @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
        """,
        (query, query, seed_k * 2),
    )
    for row in rows:
        rank = float(row[4] or 0.0)
        fulltext_candidates.append(
            {
                "id": row[0],
                "type": row[1],
                "label": row[2],
                "aliases": row[3] or [],
                "score": rank,
                "match_reason": "fulltext",
            }
        )

    # Alias exact match (cheap + good for proper nouns)
    try:
        from lib.id_generators import normalize_label

        alias_norm = normalize_label(query)
        rows = postgres.execute_query(
            """
            SELECT kn.id, kn.type, kn.label, kn.aliases
            FROM kg_aliases ka
            JOIN kg_nodes kn ON ka.node_id = kn.id
            WHERE ka.alias_norm = %s
            LIMIT 10
            """,
            (alias_norm,),
        )
        for row in rows:
            alias_candidates.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "label": row[2],
                    "aliases": row[3] or [],
                    "score": 1.0,
                    "match_reason": "alias",
                }
            )
    except Exception:
        pass

    # Dedupe each channel
    vector_deduped = _dedupe_by_id(vector_candidates)
    fulltext_deduped = _dedupe_by_id(fulltext_candidates)
    alias_deduped = _dedupe_by_id(alias_candidates)

    # Extract query intent for boosts
    intent = _extract_query_intent(query)

    # Use RRF fusion
    fused_candidates = _fuse_candidates_rrf(
        vector_deduped,
        fulltext_deduped,
        alias_deduped,
        query=query,
        k=60,
        intent=intent,
    )

    # Optionally rerank with Gemini
    if enable_rerank and fused_candidates:
        try:
            fused_candidates = _rerank_with_gemini(
                candidates=fused_candidates,
                query=query,
                model=rerank_model,
                top_n=min(50, seed_k * 3),
            )
        except Exception:
            pass

    return fused_candidates[:seed_k]


def _retrieve_edges_hops_1(
    *,
    postgres: Any,
    seed_ids: list[str],
    max_edges: int,
) -> list[dict[str, Any]]:
    if not seed_ids:
        return []
    placeholders = ",".join(["%s"] * len(seed_ids))
    rows = postgres.execute_query(
        f"""
        SELECT id, source_id, predicate, predicate_raw, target_id,
               youtube_video_id, earliest_timestamp_str, earliest_seconds,
               utterance_ids, evidence, speaker_ids, confidence
        FROM kg_edges
        WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
        ORDER BY edge_rank_score DESC NULLS LAST, confidence DESC NULLS LAST, earliest_seconds ASC
        LIMIT %s
        """,
        tuple(seed_ids + seed_ids + [max_edges]),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": row[0],
                "source_id": row[1],
                "predicate": row[2],
                "predicate_raw": row[3],
                "target_id": row[4],
                "youtube_video_id": row[5],
                "earliest_timestamp_str": row[6],
                "earliest_seconds": int(row[7] or 0),
                "utterance_ids": row[8] or [],
                "evidence": row[9],
                "speaker_ids": row[10] or [],
                "confidence": float(row[11]) if row[11] is not None else None,
            }
        )
    return out


def _hydrate_nodes(
    *,
    postgres: Any,
    node_ids: list[str],
) -> list[dict[str, Any]]:
    if not node_ids:
        return []
    placeholders = ",".join(["%s"] * len(node_ids))
    rows = postgres.execute_query(
        f"""
        SELECT id, label, type
        FROM kg_nodes
        WHERE id IN ({placeholders})
        """,
        tuple(node_ids),
    )
    return [{"id": r[0], "label": r[1], "type": r[2]} for r in rows]


def _hydrate_citations(
    *,
    postgres: Any,
    utterance_ids: list[str],
    max_citations: int,
) -> list[dict[str, Any]]:
    if not utterance_ids:
        return []
    utterance_ids = utterance_ids[:max_citations]
    placeholders = ",".join(["%s"] * len(utterance_ids))
    rows = postgres.execute_query(
        f"""
        SELECT s.id, s.text, s.seconds_since_start, s.timestamp_str,
               s.youtube_video_id, s.video_date, s.video_title, s.speaker_id,
               sp.full_name, sp.normalized_name, sp.title, sp.position,
               (
                   SELECT svr.role_label
                   FROM speaker_video_roles svr
                   WHERE svr.youtube_video_id = s.youtube_video_id
                     AND svr.speaker_id = s.speaker_id
                   ORDER BY
                     CASE svr.source
                       WHEN 'order_paper_pdf' THEN 0
                       WHEN 'order_paper' THEN 1
                       WHEN 'transcript' THEN 2
                       ELSE 3
                     END,
                     CASE svr.role_kind
                       WHEN 'executive' THEN 0
                       WHEN 'procedural' THEN 1
                       WHEN 'parliamentary' THEN 2
                       WHEN 'constituency' THEN 3
                       WHEN 'committee' THEN 4
                       ELSE 5
                     END,
                     length(svr.role_label) ASC
                   LIMIT 1
               ) AS speaker_title
        FROM sentences s
        LEFT JOIN speakers sp ON s.speaker_id = sp.id
        WHERE s.id IN ({placeholders})
        """,
        tuple(utterance_ids),
    )

    order_paper_idx = _load_order_paper_speaker_index(postgres=postgres)
    out: list[dict[str, Any]] = []
    for r in rows:
        seconds = int(r[2] or 0)
        youtube_video_id = r[4]
        speaker_id = r[7]
        full_name = r[8]
        normalized_name = r[9]
        speaker_title = r[10]
        speaker_position = r[11]
        session_speaker_title = r[12]

        # Build key candidates for order-paper lookup.
        candidates: list[str] = []
        for raw in [full_name, normalized_name, _speaker_id_to_name_guess(speaker_id)]:
            norm = normalize_label(str(raw or ""))
            if not norm:
                continue
            candidates.append(norm)
            stripped = _strip_honorific_prefix(norm)
            if stripped and stripped != norm:
                candidates.append(stripped)

        speaker_name = ""
        for key in candidates:
            if key in order_paper_idx:
                speaker_name = order_paper_idx[key]
                break

        if not speaker_name:
            # Fall back to speakers table fields.
            base_full = str(full_name or "").strip()
            base_norm = str(normalized_name or "").strip()

            # If full_name is missing or looks unformatted, prefer normalized formatting.
            looks_unformatted = (
                not base_full or base_full == base_full.lower() or base_full.startswith("s_")
            )
            if looks_unformatted:
                # If the stored normalized_name is actually the speaker_id,
                # derive a readable name from the id itself.
                norm_source = base_norm
                if base_norm.startswith("s_"):
                    norm_source = _speaker_id_to_name_guess(base_norm)
                elif speaker_id and str(speaker_id).startswith("s_"):
                    norm_source = _speaker_id_to_name_guess(str(speaker_id))

                speaker_name = format_speaker_name(
                    full_name=None,
                    normalized_name=norm_source or base_full,
                    speaker_id=str(speaker_id or ""),
                )
            else:
                speaker_name = base_full

            # If a title exists separately, prefix when missing.
            if speaker_title and "honourable" not in speaker_name.lower():
                speaker_name = _format_title_and_name(str(speaker_title), speaker_name)

        out.append(
            {
                "utterance_id": r[0],
                "text": r[1],
                "seconds_since_start": seconds,
                "timestamp_str": r[3] or "",
                "youtube_video_id": youtube_video_id,
                "youtube_url": build_youtube_url(youtube_video_id, seconds),
                "video_date": str(r[5]) if r[5] else None,
                "video_title": r[6] or None,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "speaker_title": (
                    str(session_speaker_title)
                    if session_speaker_title
                    else (
                        str(speaker_position)
                        if speaker_position and str(speaker_position).strip().lower() != "unknown"
                        else None
                    )
                ),
            }
        )
    out.sort(key=lambda x: int(x.get("seconds_since_start", 0)))
    return out


def kg_hybrid_graph_rag(
    *,
    postgres: Any,
    embedding_client: Any,
    query: str,
    hops: int = 1,
    seed_k: int = 12,
    max_edges: int = 90,
    max_citations: int = 12,
    edge_rank_threshold: float | None = None,
    query_embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Hybrid retrieve seeds (vector+FTS), then expand KG edges N hops.

    This is designed to be used as a deterministic tool for an agent loop.

    Args:
        postgres: Database client
        embedding_client: Embedding generation client
        query: Search query
        hops: Number of graph hops to expand (default: 1)
        seed_k: Number of seed nodes to retrieve (default: 12)
        max_edges: Maximum edges to return (default: 90)
        max_citations: Maximum citations to return (default: 12)
        edge_rank_threshold: Optional threshold to filter edges by edge_rank_score.
            Only returns edges with score >= threshold.
            Recommended: 0.05 after normalization, or 0.00001 before normalization.
            None means no threshold filtering (default: None)
    """
    query = (query or "").strip()
    if not query:
        return {
            "query": "",
            "hops": int(hops),
            "seeds": [],
            "nodes": [],
            "edges": [],
            "citations": [],
            "debug": {"reason": "empty_query"},
        }

    hops = max(1, int(hops))
    seed_k = max(1, int(seed_k))
    max_edges = max(1, int(max_edges))
    max_citations = max(1, int(max_citations))

    # Get reranker config
    try:
        from lib.utils.config import config

        enable_rerank = getattr(config, "enable_seed_rerank", False)
        rerank_model = getattr(config, "seed_rerank_model", "gemini-2.0-flash")
    except Exception:
        enable_rerank = False
        rerank_model = "gemini-2.0-flash"

    seeds = _retrieve_seed_nodes(
        postgres=postgres,
        embedding_client=embedding_client,
        query=query,
        seed_k=seed_k,
        enable_rerank=enable_rerank,
        rerank_model=rerank_model,
        query_embedding=query_embedding,
    )
    seed_ids = [s["id"] for s in seeds]

    # Start with hop=1 (dominant use case). Hops>1 can be layered later.
    edges: list[dict[str, Any]] = []
    frontier_ids = seed_ids
    seen_node_ids: set[str] = set(seed_ids)

    if hops >= 1 and frontier_ids:
        hop_edges = _retrieve_edges_hops_1(
            postgres=postgres,
            seed_ids=frontier_ids,
            max_edges=max_edges,
        )
        edges.extend(hop_edges)
        for e in hop_edges:
            seen_node_ids.add(e["source_id"])
            seen_node_ids.add(e["target_id"])

    # If hops>1, expand iteratively from newly discovered nodes
    # (kept intentionally conservative to avoid blowups)
    for _hop in range(2, hops + 1):
        next_frontier = [nid for nid in seen_node_ids if nid not in set(frontier_ids)]
        if not next_frontier:
            break
        frontier_ids = next_frontier
        hop_edges = _retrieve_edges_hops_1(
            postgres=postgres,
            seed_ids=frontier_ids,
            max_edges=max_edges,
        )
        edges.extend(hop_edges)
        for e in hop_edges:
            seen_node_ids.add(e["source_id"])
            seen_node_ids.add(e["target_id"])
        if len(edges) >= max_edges:
            edges = edges[:max_edges]
            break

    nodes = _hydrate_nodes(postgres=postgres, node_ids=sorted(seen_node_ids))
    node_by_id = {n.get("id"): n for n in nodes}
    for e in edges:
        source = node_by_id.get(e.get("source_id"), {})
        target = node_by_id.get(e.get("target_id"), {})
        e["source_label"] = source.get("label")
        e["source_type"] = source.get("type")
        e["target_label"] = target.get("label")
        e["target_type"] = target.get("type")

    utterance_ids: list[str] = []
    for e in edges:
        for uid in e.get("utterance_ids", []) or []:
            if uid not in utterance_ids:
                utterance_ids.append(uid)

    edges_filtered: int = 0
    if edge_rank_threshold is not None:
        edges_before_filter = len(edges)
        edges = [e for e in edges if e.get("edge_rank_score", 0.0) >= edge_rank_threshold]
        edges_filtered = edges_before_filter - len(edges)
        if edges_filtered > 0:
            edges = edges[:max_edges]

    citations = _hydrate_citations(
        postgres=postgres,
        utterance_ids=utterance_ids,
        max_citations=max_citations,
    )

    debug_info: dict[str, Any] = {
        "seed_count": len(seeds),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "citation_count": len(citations),
    }
    if edge_rank_threshold is not None:
        debug_info = {
            **debug_info,
            "edge_rank_threshold": float(edge_rank_threshold),
            "edges_filtered_by_threshold": edges_filtered,
        }

    return {
        "query": query,
        "hops": hops,
        "seeds": seeds,
        "nodes": nodes,
        "edges": edges,
        "citations": citations,
        "debug": debug_info,
    }


def _retrieve_bill_excerpts(
    *,
    postgres: Any,
    embedding_client: Any,
    query: str,
    seed_bill_ids: list[str] | None = None,
    max_bill_citations: int = 8,
    min_bill_score: float = 0.35,
    max_chunks_per_bill: int = 1,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve bill excerpts matching the query using hybrid vector + BM25 search.

    Args:
        postgres: Database client
        embedding_client: Embedding generation client
        query: Search query
        seed_bill_ids: Optional list of bill IDs to boost in results
        max_bill_citations: Maximum number of bill excerpts to return

    Returns:
        List of bill excerpt citations with bill metadata
    """
    if not query or not query.strip():
        return []

    try:
        embedding = query_embedding
        if embedding is None:
            embedding = embedding_client.generate_query_embedding(query)
    except Exception:
        return []

    seed_bill_set = set(seed_bill_ids or [])
    query_terms = _query_terms(query)

    sql = """
        SELECT be.id, be.bill_id, be.chunk_index, be.text, be.source_url,
               be.embedding <=> (%s::vector) AS distance,
               b.bill_number, b.title,
               be.page_number,
               ts_rank_cd(be.tsv, plainto_tsquery('english', %s)) AS bm25_rank
        FROM bill_excerpts be
        JOIN bills b ON be.bill_id = b.id
        WHERE be.embedding IS NOT NULL
        ORDER BY distance ASC, bm25_rank DESC, be.chunk_index ASC
        LIMIT %s
    """

    try:
        candidate_limit = max(max_bill_citations * 8, 24)
        params = [vector_literal(embedding), query, candidate_limit]
        rows = postgres.execute_query(sql, params)
    except Exception:
        rows = []

    out: list[dict[str, Any]] = []
    per_bill_counts: dict[str, int] = {}
    for row in rows:
        distance = float(row[5] or 1.0)
        vector_score = max(0.0, 1.0 - distance)
        bm25_rank = float(row[9] or 0.0) if len(row) > 9 else 0.0
        bm25_score = min(1.0, bm25_rank)

        bill_id = str(row[1] or "")
        boost = 0.08 if bill_id in seed_bill_set else 0.0
        score = (0.75 * vector_score) + (0.25 * bm25_score) + boost
        if score < min_bill_score:
            continue

        excerpt = str(row[3] or "")
        excerpt_lower = excerpt.lower()
        matched_terms = [term for term in query_terms if term in excerpt_lower]
        if query_terms and not matched_terms:
            continue

        if per_bill_counts.get(bill_id, 0) >= max_chunks_per_bill:
            continue
        per_bill_counts[bill_id] = per_bill_counts.get(bill_id, 0) + 1

        citation_id = f"bill:{bill_id}:{row[2]}"
        page_number = int(row[8] or 0) if len(row) > 8 and row[8] else None
        source_url = _url_with_page_fragment(str(row[4] or ""), page_number)

        out.append(
            {
                "citation_id": citation_id,
                "bill_id": bill_id,
                "bill_number": row[6] or "",
                "bill_title": row[7] if len(row) > 7 else "",
                "excerpt": excerpt,
                "source_url": source_url,
                "chunk_index": row[2],
                "page_number": page_number,
                "matched_terms": matched_terms[:6],
                "score": score,
            }
        )

        if len(out) >= max_bill_citations:
            break

    return out


def kg_hybrid_graph_rag_with_bills(
    *,
    postgres: Any,
    embedding_client: Any,
    query: str,
    hops: int = 1,
    seed_k: int = 12,
    max_edges: int = 90,
    max_citations: int = 12,
    max_bill_citations: int = 8,
    edge_rank_threshold: float | None = None,
) -> dict[str, Any]:
    """Hybrid Graph-RAG with bill excerpt retrieval.

    Extends kg_hybrid_graph_rag to include bill excerpt citations.

    Args:
        postgres: Database client
        embedding_client: Embedding generation client
        query: Search query
        hops: Number of graph hops to expand
        seed_k: Number of seed nodes to retrieve
        max_edges: Maximum edges to return
        max_citations: Maximum transcript citations to return
        max_bill_citations: Maximum bill excerpt citations to return
        edge_rank_threshold: Optional threshold to filter edges

    Returns:
        Dict with seeds, nodes, edges, citations, and bill_citations
    """
    query_embedding: list[float] | None = None
    try:
        query_embedding = embedding_client.generate_query_embedding(query)
    except Exception:
        query_embedding = None

    result = kg_hybrid_graph_rag(
        postgres=postgres,
        embedding_client=embedding_client,
        query=query,
        hops=hops,
        seed_k=seed_k,
        max_edges=max_edges,
        max_citations=max_citations,
        edge_rank_threshold=edge_rank_threshold,
        query_embedding=query_embedding,
    )

    seed_bill_ids: list[str] = []
    for seed in result.get("seeds", []):
        if seed.get("type") == "BILL" or seed.get("type") == "schema:Legislation":
            seed_bill_ids.append(seed.get("id", ""))

    bill_citations = _retrieve_bill_excerpts(
        postgres=postgres,
        embedding_client=embedding_client,
        query=query,
        seed_bill_ids=seed_bill_ids,
        max_bill_citations=max_bill_citations,
        query_embedding=query_embedding,
    )

    result["bill_citations"] = bill_citations
    result["debug"]["bill_citation_count"] = len(bill_citations)

    return result
