from __future__ import annotations

from typing import Any

from lib.db.pgvector import vector_literal


def build_youtube_url(youtube_video_id: str, seconds_since_start: int) -> str:
    seconds = int(seconds_since_start)
    if seconds < 0:
        seconds = 0
    return f"https://www.youtube.com/watch?v={youtube_video_id}&t={seconds}s"


def _dedupe_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("id", ""))
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        out.append(item)
    return out


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
        ORDER BY confidence DESC NULLS LAST, earliest_seconds ASC
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
               sp.full_name, sp.normalized_name
        FROM sentences s
        LEFT JOIN speakers sp ON s.speaker_id = sp.id
        WHERE s.id IN ({placeholders})
        """,
        tuple(utterance_ids),
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        seconds = int(r[2] or 0)
        youtube_video_id = r[4]
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
                "speaker_id": r[7],
                "speaker_name": (r[9] or r[8] or r[7] or "").strip(),
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
    seed_k: int = 8,
    max_edges: int = 60,
    max_citations: int = 12,
) -> dict[str, Any]:
    """Hybrid retrieve seeds (vector+FTS), then expand KG edges N hops.

    This is designed to be used as a deterministic tool for an agent loop.
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

    utterance_ids: list[str] = []
    for e in edges:
        for uid in e.get("utterance_ids", []) or []:
            if uid not in utterance_ids:
                utterance_ids.append(uid)
    citations = _hydrate_citations(
        postgres=postgres,
        utterance_ids=utterance_ids,
        max_citations=max_citations,
    )

    return {
        "query": query,
        "hops": hops,
        "seeds": seeds,
        "nodes": nodes,
        "edges": edges,
        "citations": citations,
        "debug": {
            "seed_count": len(seeds),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "citation_count": len(citations),
        },
    }
