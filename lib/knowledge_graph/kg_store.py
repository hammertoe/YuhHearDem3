"""Shared storage and canonicalization for knowledge graph extraction.

This module contains storage logic that is independent of the LLM provider.
It can be used with Gemini, Cerebras, or any other LLM-based extractor.
"""

from __future__ import annotations

from typing import Any

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.id_generators import generate_kg_edge_id, generate_kg_node_id, normalize_label
from lib.db.pgvector import vector_literal
from lib.knowledge_graph.window_builder import Window


def canonicalize_and_store(
    *,
    postgres: PostgresClient,
    embedding: GoogleEmbeddingClient,
    results: list[
        tuple[Window, list[dict[str, Any]], list[dict[str, Any]], str, bool, str | None]
    ],
    youtube_video_id: str,
    kg_run_id: str,
    extractor_model: str,
) -> dict[str, Any]:
    """Canonicalize nodes and edges and store them in Postgres.

    Args:
        postgres: Postgres client for database operations
        embedding: Google embedding client for generating embeddings
        results: List of (window, nodes_new, edges, raw_response, parse_success, error)
        youtube_video_id: YouTube video ID
        kg_run_id: Unique ID for this extraction run
        extractor_model: Name of the LLM model used

    Returns:
        Dictionary with statistics about the operation
    """

    def _normalize_speaker_ref(ref: str, window_speaker_ids: list[str]) -> str | None:
        ref = (ref or "").strip()
        if not ref:
            return None

        if ref.startswith("speaker_"):
            sid = ref.removeprefix("speaker_")
            return ref if sid in window_speaker_ids else None

        if ref.startswith("s_"):
            return f"speaker_{ref}" if ref in window_speaker_ids else None

        return ref

    temp_to_canonical = {}
    new_nodes_data = []
    new_aliases_data = []
    edges_data = []
    stats = {
        "windows_processed": len(results),
        "windows_successful": 0,
        "windows_failed": 0,
        "new_nodes": 0,
        "edges": 0,
        "links_to_known": 0,
        "edges_skipped_invalid_speaker_ref": 0,
        "edges_skipped_missing_nodes": 0,
    }

    # Ensure speaker nodes exist for all speakers observed in successful windows.
    speaker_ids_seen: set[str] = set()
    for result in results:
        window = result[0]
        parse_success = result[4]
        if parse_success:
            speaker_ids_seen.update(window.speaker_ids)

    if speaker_ids_seen:
        speaker_rows = postgres.execute_query(
            """
            SELECT id, normalized_name, full_name, title
            FROM speakers
            WHERE id = ANY(%s)
            """,
            (list(speaker_ids_seen),),
        )
        speaker_meta = {
            row[0]: {
                "normalized_name": row[1] or "",
                "full_name": row[2] or "",
                "title": row[3] or "",
            }
            for row in speaker_rows
        }

        speaker_nodes_data = []
        for speaker_id in speaker_ids_seen:
            meta = speaker_meta.get(speaker_id, {})
            label = meta.get("full_name") or meta.get("normalized_name") or speaker_id
            aliases = []
            for candidate in (
                meta.get("full_name"),
                meta.get("normalized_name"),
                meta.get("title"),
                speaker_id,
            ):
                if candidate:
                    aliases.append(normalize_label(candidate))

            speaker_nodes_data.append(
                (
                    f"speaker_{speaker_id}",
                    label,
                    "foaf:Person",
                    aliases,
                )
            )

        node_query = """
            INSERT INTO kg_nodes (id, label, type, aliases)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET label = EXCLUDED.label,
                aliases = EXCLUDED.aliases,
                updated_at = NOW()
        """
        postgres.execute_batch(node_query, speaker_nodes_data)

    for result in results:
        window = result[0]
        nodes_new = result[1]
        edges_list = result[2]
        parse_success = result[4]

        if not parse_success:
            stats["windows_failed"] += 1
            continue

        stats["windows_successful"] += 1

        utterance_timestamps = {
            u.id: (u.timestamp_str, u.seconds_since_start) for u in window.utterances
        }

        for node in nodes_new:
            node_id = generate_kg_node_id(node["type"], node["label"])

            temp_to_canonical[node["temp_id"]] = node_id

            new_nodes_data.append(
                (
                    node_id,
                    node["label"],
                    node["type"],
                    node.get("aliases", []),
                )
            )

            for alias in node.get("aliases", []):
                if alias:
                    new_aliases_data.append(
                        (
                            normalize_label(alias),
                            alias,
                            node_id,
                            node["type"],
                            "llm",
                            None,
                        )
                    )

            stats["new_nodes"] += 1

        for edge in edges_list:
            window_speaker_ids = window.speaker_ids

            source_ref = _normalize_speaker_ref(edge["source_ref"], window_speaker_ids)
            target_ref = _normalize_speaker_ref(edge["target_ref"], window_speaker_ids)

            if source_ref is None or target_ref is None:
                stats["edges_skipped_invalid_speaker_ref"] += 1
                continue

            source_id = temp_to_canonical.get(source_ref, source_ref)
            target_id = temp_to_canonical.get(target_ref, target_ref)

            if not (
                edge["source_ref"].startswith("speaker_")
                or edge["source_ref"] in temp_to_canonical
            ):
                stats["links_to_known"] += 1
            if not (
                edge["target_ref"].startswith("speaker_")
                or edge["target_ref"] in temp_to_canonical
            ):
                stats["links_to_known"] += 1

            utterance_ids = edge.get("utterance_ids", [])
            earliest_timestamp_str = None
            earliest_seconds = None

            for uid in utterance_ids:
                if uid in utterance_timestamps:
                    ts_str, ts_seconds = utterance_timestamps[uid]
                    if earliest_seconds is None or ts_seconds < earliest_seconds:
                        earliest_timestamp_str = ts_str
                        earliest_seconds = ts_seconds

            edge_id = generate_kg_edge_id(
                source_id,
                edge["predicate"],
                target_id,
                youtube_video_id,
                earliest_seconds or window.earliest_seconds or 0,
                edge["evidence"],
            )

            edges_data.append(
                (
                    edge_id,
                    source_id,
                    edge["predicate"],
                    target_id,
                    youtube_video_id,
                    earliest_timestamp_str or window.earliest_timestamp,
                    earliest_seconds or window.earliest_seconds,
                    utterance_ids,
                    edge["evidence"],
                    window.speaker_ids,
                    float(edge.get("confidence", 0.5)),
                    extractor_model,
                    kg_run_id,
                )
            )

            stats["edges"] += 1

    if new_nodes_data:
        node_query = """
            INSERT INTO kg_nodes (id, label, type, aliases)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET label = EXCLUDED.label,
                aliases = EXCLUDED.aliases,
                updated_at = NOW()
        """
        postgres.execute_batch(node_query, new_nodes_data)

    if new_aliases_data:
        alias_query = """
            INSERT INTO kg_aliases (alias_norm, alias_raw, node_id, type, source, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (alias_norm) DO NOTHING
        """
        postgres.execute_batch(alias_query, new_aliases_data)

    if edges_data:
        # Drop edges whose endpoints do not exist; this prevents a single bad edge
        # from failing the entire run.
        referenced_node_ids = sorted(
            {src for (_eid, src, *_rest) in edges_data}
            | {tgt for (_eid, _src, _pred, tgt, *_rest) in edges_data}
        )

        existing_rows = postgres.execute_query(
            """
            SELECT id
            FROM kg_nodes
            WHERE id = ANY(%s)
            """,
            (referenced_node_ids,),
        )
        existing_ids = {row[0] for row in existing_rows}

        filtered_edges = [
            e for e in edges_data if e[1] in existing_ids and e[3] in existing_ids
        ]
        stats["edges_skipped_missing_nodes"] = len(edges_data) - len(filtered_edges)
        stats["edges"] = len(filtered_edges)

        edge_query = """
            INSERT INTO kg_edges (
                id, source_id, predicate, target_id, youtube_video_id,
                earliest_timestamp_str, earliest_seconds, utterance_ids,
                evidence, speaker_ids, confidence, extractor_model, kg_run_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """
        if filtered_edges:
            postgres.execute_batch(edge_query, filtered_edges)

    # Generate embeddings for newly created nodes.
    if new_nodes_data:
        _embed_new_nodes(postgres, embedding, [nid for (nid, *_rest) in new_nodes_data])

    return stats


def _embed_new_nodes(
    postgres: PostgresClient,
    embedding: GoogleEmbeddingClient,
    node_ids: list[str],
) -> None:
    """Generate embeddings for nodes that don't have them yet."""
    if not node_ids:
        return

    rows = postgres.execute_query(
        """
        SELECT id, label
        FROM kg_nodes
        WHERE id = ANY(%s) AND embedding IS NULL
        """,
        (node_ids,),
    )

    to_embed = [(row[0], row[1]) for row in rows if row[1]]
    if not to_embed:
        return

    ids = [x[0] for x in to_embed]
    texts = [x[1] for x in to_embed]
    embeddings = embedding.generate_embeddings_batch(
        texts, task_type="RETRIEVAL_DOCUMENT"
    )

    update_rows = [
        (vector_literal(vec), node_id) for node_id, vec in zip(ids, embeddings)
    ]
    postgres.execute_batch(
        """
        UPDATE kg_nodes
        SET embedding = (%s)::vector, updated_at = NOW()
        WHERE id = %s
        """,
        update_rows,
    )
