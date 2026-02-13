from __future__ import annotations

import re
from typing import Any

from lib.db.pgvector import vector_literal
from lib.id_generators import normalize_label


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


def _retrieve_seed_nodes(
    *,
    postgres: Any,
    embedding_client: Any,
    query: str,
    seed_k: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # Vector search (best when embeddings exist)
    try:
        query_embedding = embedding_client.generate_query_embedding(query)
        rows = postgres.execute_query(
            """
            SELECT id, type, label, aliases, embedding <=> %s AS distance
            FROM kg_nodes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s ASC
            LIMIT %s
            """,
            (vector_literal(query_embedding), vector_literal(query_embedding), seed_k),
        )
        for row in rows:
            distance = float(row[4] or 0.0)
            candidates.append(
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
        # Embeddings optional; full-text below still works.
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
        (query, query, seed_k),
    )
    for row in rows:
        rank = float(row[4] or 0.0)
        candidates.append(
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
            LIMIT 5
            """,
            (alias_norm,),
        )
        for row in rows:
            candidates.append(
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

    # Dedupe and keep best scores first
    deduped = _dedupe_by_id(candidates)
    deduped.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return deduped[:seed_k]


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

    seeds = _retrieve_seed_nodes(
        postgres=postgres,
        embedding_client=embedding_client,
        query=query,
        seed_k=seed_k,
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

    debug_info = {
        "seed_count": len(seeds),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "citation_count": len(citations),
    }
    if edge_rank_threshold is not None:
        debug_info["edge_rank_threshold"] = edge_rank_threshold
        debug_info["edges_filtered_by_threshold"] = edges_filtered

    return {
        "query": query,
        "hops": hops,
        "seeds": seeds,
        "nodes": nodes,
        "edges": edges,
        "citations": citations,
        "debug": debug_info,
    }
