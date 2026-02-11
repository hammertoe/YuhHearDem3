"""Advanced search features: temporal search, trends, multi-hop queries."""

from __future__ import annotations

import argparse
import json
from typing import Any
from datetime import datetime

from lib.db.postgres_client import PostgresClient
from lib.db.memgraph_client import MemgraphClient
from lib.embeddings.google_client import GoogleEmbeddingClient


class AdvancedSearchFeatures:
    """Advanced search features beyond basic hybrid search."""

    def __init__(
        self,
        postgres: PostgresClient | None = None,
        memgraph: MemgraphClient | None = None,
        embedding_client: GoogleEmbeddingClient | None = None,
    ):
        self.postgres = postgres or PostgresClient()
        self.memgraph = memgraph
        if self.memgraph is None:
            try:
                self.memgraph = MemgraphClient()
            except Exception:
                print("Warning: Could not connect to Memgraph, graph features will be disabled")
                self.memgraph = None
        self.embedding_client = embedding_client or GoogleEmbeddingClient()

    def temporal_search(
        self,
        query: str,
        date_start: str | None = None,
        date_end: str | None = None,
        speaker_id: str | None = None,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search sentences using paragraph embeddings with optional filters."""
        print(f"\nTemporal search for: '{query}'")
        print(f"Date range: {date_start or 'Start'} to {date_end or 'End'}")
        print(f"Speaker filter: {speaker_id or 'All'}")
        print(f"Entity type filter: {entity_type or 'All'}")
        print(f"Limit: {limit}")

        query_embedding = self.embedding_client.generate_query_embedding(query)

        where_clauses = ["1=1"]
        where_params: list[Any] = []

        if date_start:
            where_clauses.append("p.video_date >= %s")
            where_params.append(date_start)

        if date_end:
            where_clauses.append("p.video_date <= %s")
            where_params.append(date_end)

        if speaker_id:
            where_clauses.append("p.speaker_id = %s")
            where_params.append(speaker_id)

        # Optional filter: only sentences that have a matching entity_type.
        join_entity_filter = ""
        join_params: list[Any] = []
        if entity_type:
            join_entity_filter = (
                "JOIN sentence_entities se ON se.sentence_id = s.id AND se.entity_type = %s"
            )
            join_params.append(entity_type)

        params: list[Any] = [query_embedding, *join_params, *where_params, limit]

        sql = f"""
            SELECT
                s.id,
                s.text,
                s.seconds_since_start,
                s.timestamp_str,
                p.youtube_video_id,
                p.video_title,
                p.video_date,
                p.speaker_id,
                sp.normalized_name AS speaker_name,
                p.id AS paragraph_id,
                p.embedding <=> %s AS distance
            FROM paragraphs p
            JOIN sentences s ON s.paragraph_id = p.id AND s.sentence_order = 1
            LEFT JOIN speakers sp ON p.speaker_id = sp.id
            {join_entity_filter}
            WHERE {" AND ".join(where_clauses)}
            ORDER BY distance ASC
            LIMIT %s
        """

        results = self.postgres.execute_query(sql, tuple(params))

        formatted: list[dict[str, Any]] = []
        for row in results:
            distance = float(row[10] or 0)
            formatted.append(
                {
                    "id": row[0],
                    "text": row[1],
                    "seconds_since_start": row[2],
                    "timestamp_str": row[3],
                    "video_id": row[4],
                    "video_title": row[5] or "",
                    "video_date": str(row[6]) if row[6] is not None else "",
                    "speaker_id": row[7],
                    "speaker_name": row[8] or row[7],
                    "paragraph_id": row[9],
                    "score": 1.0 - distance,
                    "search_type": "temporal",
                }
            )

        print(f"✅ Found {len(formatted)} results")
        return formatted

    def trend_analysis(
        self,
        entity_id: str | None = None,
        time_window_days: int = 30,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Analyze mention trends over time."""
        print(f"\nTrend analysis for entity: {entity_id or 'All'}")
        print(f"Time window: {time_window_days} days")

        params: list[Any] = [time_window_days]
        entity_clause = ""
        if entity_id:
            entity_clause = "AND se.entity_id = %s"
            params.append(entity_id)

        params.append(limit)

        sql = f"""
            SELECT
                DATE(s.video_date) AS date,
                COUNT(DISTINCT s.id) AS mentions
            FROM sentence_entities se
            JOIN sentences s ON se.sentence_id = s.id
            WHERE s.video_date >= CURRENT_DATE - (%s * INTERVAL '1 day')
            {entity_clause}
            GROUP BY DATE(s.video_date)
            ORDER BY date ASC
            LIMIT %s
        """

        results = self.postgres.execute_query(sql, tuple(params))

        daily_mentions = [{"date": row[0], "mentions": row[1]} for row in results]

        if not daily_mentions:
            return {
                "entity_id": entity_id,
                "trends": [],
                "summary": {
                    "total_mentions": 0,
                    "date_range": "",
                    "average_daily": 0.0,
                    "peak_date": "",
                    "peak_mentions": 0,
                },
                "moving_average": [],
            }

        moving_avg = self._calculate_moving_average(daily_mentions)

        total_mentions = sum(m["mentions"] for m in daily_mentions)
        peak = max(daily_mentions, key=lambda x: x["mentions"])

        return {
            "entity_id": entity_id,
            "trends": daily_mentions,
            "summary": {
                "total_mentions": total_mentions,
                "date_range": f"{daily_mentions[0]['date']} to {daily_mentions[-1]['date']}",
                "average_daily": total_mentions / len(daily_mentions),
                "peak_date": peak["date"],
                "peak_mentions": peak["mentions"],
            },
            "moving_average": moving_avg,
        }

    def _calculate_moving_average(
        self, data: list[dict[str, Any]], window_size: int = 7
    ) -> list[dict[str, Any]]:
        """Calculate moving average for trend smoothing."""
        moving_avg = []

        for i in range(len(data)):
            if i < window_size - 1:
                window = data[: i + 1]
                avg = sum(d["mentions"] for d in window) / len(window)
            else:
                window = data[max(0, i - window_size + 1) : i + 1]
                avg = sum(d["mentions"] for d in window) / len(window)

            moving_avg.append({"date": data[i]["date"], "value": avg})

        return moving_avg

    def multi_hop_query(
        self,
        start_entity_id: str,
        hops: int = 2,
        max_results: int = 50,
        relationship_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute multi-hop graph query."""
        print(f"\nMulti-hop query from: {start_entity_id}")
        print(f"Hops: {hops}")
        print(f"Max results: {max_results}")

        if not relationship_types:
            relationship_types = [
                "DISCUSSES",
                "MENTIONS",
                "AGREES_WITH",
                "DISAGREES_WITH",
                "QUESTIONS",
                "RESPONDS_TO",
                "ADVOCATES_FOR",
                "CRITICIZES",
                "WORKS_WITH",
            ]

        rel_types_str = ", ".join(f"'{rt}'" for rt in relationship_types)

        cypher = f"""
            MATCH (start {{id: $start_id}})-[r*1..{hops}]-(related)
            WHERE any(rel IN r WHERE type(rel) IN [{rel_types_str}])
            RETURN DISTINCT
                start.id as start_entity,
                related.id as related_entity,
                type(last(r)) as relationship_type,
                labels(related)[0] as related_type
            LIMIT $limit
        """

        results = self.memgraph.execute_query(
            cypher, {"start_id": start_entity_id, "limit": max_results}
        )

        print(f"✅ Multi-hop query found {len(results)} relationships")

        return results

    def complex_query(self, query_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute complex graph queries."""
        print(f"\nComplex query: {query_type}")
        print(f"Params: {params}")

        queries = {
            "speaker_influence": """
                MATCH (s:Speaker)-[r:DISCUSSES]->(topic:Topic)
                WITH s, topic, COUNT(r) AS count
                RETURN s.id AS speaker_id,
                       s.normalized_name AS speaker_name,
                       topic.id AS topic_id,
                       topic.text AS topic_text,
                       count
                ORDER BY count DESC
                LIMIT 20
            """,
            "bill_connections": """
                MATCH (b:Bill)-[r:DISCUSSED_BY]->(s:Speaker)
                RETURN DISTINCT
                    b.id AS bill_id,
                    b.title AS bill_title,
                    b.status AS bill_status,
                    COUNT(r) AS speaker_count
                ORDER BY speaker_count DESC
                LIMIT 20
            """,
            "controversial_topics": """
                MATCH (s1:Speaker)-[r:DISAGREES_WITH]->(s2:Speaker)
                MATCH (s1)-[r:DISCUSSES]->(topic:Topic)<-[r:AGREES_WITH]-(s2)
                RETURN DISTINCT
                    topic.text AS topic,
                    COUNT(r) AS agree_count,
                    s1.id AS disagreeing_speaker_1,
                    s1.name AS disagreeing_speaker_name_1,
                    s2.id AS disagreeing_speaker_2,
                    s2.name AS disagreeing_speaker_name_2
                ORDER BY agree_count DESC
                LIMIT 50
            """,
        }

        query = queries.get(query_type)
        if not query:
            return {"query_type": query_type, "results": [], "count": 0}

        results = self.memgraph.execute_query(query, params or {})
        output = {"query_type": query_type, "results": results, "count": len(results)}

        print(f"✅ Query found {output['count']} results")
        return output


def main():
    parser = argparse.ArgumentParser(description="Advanced search features")
    subparsers = parser.add_subparsers(dest="command")

    temporal_parser = subparsers.add_parser("temporal")
    temporal_parser.add_argument("--query", required=True)
    temporal_parser.add_argument("--date-start")
    temporal_parser.add_argument("--date-end")
    temporal_parser.add_argument("--speaker-id")
    temporal_parser.add_argument("--limit", type=int, default=20)

    trends_parser = subparsers.add_parser("trends")
    trends_parser.add_argument("--entity-id")
    trends_parser.add_argument("--window-days", type=int, default=30)

    multi_hop_parser = subparsers.add_parser("multi-hop")
    multi_hop_parser.add_argument("--entity-id", required=True)
    multi_hop_parser.add_argument("--hops", type=int, default=2)
    multi_hop_parser.add_argument("--max-results", type=int, default=50)
    multi_hop_parser.add_argument("--relationship-types", nargs="+")

    complex_parser = subparsers.add_parser("complex")
    complex_parser.add_argument(
        "--query-type",
        required=True,
        choices=["speaker_influence", "bill_connections", "controversial_topics"],
    )
    complex_parser.add_argument("--max-results", type=int, default=20)

    args = parser.parse_args()

    print("=" * 80)
    print("Advanced Search Features - Phase 4")
    print("=" * 80)

    features = AdvancedSearchFeatures()

    if args.command == "temporal":
        results = features.temporal_search(
            args.query,
            args.date_start,
            args.date_end,
            args.speaker_id,
            None,
            args.limit,
        )
        print(f"\n✅ Temporal search returned {len(results)} results")

        with open("temporal_search_results.json", "w") as f:
            json.dump(
                {
                    "query": args.query,
                    "filters": {
                        "date_start": args.date_start,
                        "date_end": args.date_end,
                        "speaker_id": args.speaker_id,
                    },
                    "results": results,
                    "search_timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    elif args.command == "trends":
        results = features.trend_analysis(args.entity_id, args.window_days, args.limit)
        print("\n✅ Trend analysis complete")
        with open("trend_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)

    elif args.command == "multi-hop":
        results = features.multi_hop_query(
            args.entity_id, args.hops, args.max_results, args.relationship_types
        )
        print("\n✅ Multi-hop query complete")
        with open("multi_hop_results.json", "w") as f:
            json.dump(results, f, indent=2)

    elif args.command == "complex":
        results = features.complex_query(args.query_type, {"max_results": args.max_results})
        print("\n✅ Complex query complete")
        with open("complex_search_results.json", "w") as f:
            json.dump(results, f, indent=2)

    else:
        parser.print_help()

    print("=" * 80)
    print("✅ Search complete!")
