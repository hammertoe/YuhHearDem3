"""Memgraph connection manager for graph storage."""

from __future__ import annotations

from typing import Any

from mgclient import connect, Node, Relationship
from lib.utils.config import config


class MemgraphClient:
    """Memgraph connection manager."""

    def __init__(self):
        self.client: Any
        self._connect()

    def _connect(self):
        """Connect to Memgraph."""
        host = config.database.memgraph_host
        port = config.database.memgraph_port

        auth: tuple[str, str] | None = None
        if config.database.memgraph_user and config.database.memgraph_password:
            auth = (config.database.memgraph_user, config.database.memgraph_password)

        self.client = connect(host=host, port=port)
        if auth:
            self.client.authenticate(*auth)

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results."""
        cursor = self.client.cursor()
        cursor.execute(query, params or {})

        results = []
        if cursor.description:
            columns = []
            for column in cursor.description:
                name = getattr(column, "name", None)
                if name is None:
                    try:
                        name = column[0]
                    except Exception:
                        name = str(column)
                columns.append(name)
        else:
            columns = []
        rows = cursor.fetchall() if hasattr(cursor, "fetchall") else []
        for row in rows:
            results.append(dict(zip(columns, row)))
        return results

    def execute_update(self, query: str, params: dict[str, Any] | None = None) -> int:
        """Execute an update query and return affected rows."""
        cursor = self.client.cursor()
        cursor.execute(query, params or {})
        return cursor.rowcount if cursor.rowcount >= 0 else 0

    def create_entity(self, label: str, properties: dict[str, Any]) -> Node:
        """Create an entity node."""
        query = f"""
            CREATE (e:{label})
            SET e = $properties
            RETURN e
        """
        result = self.execute_query(query, {"properties": properties})
        return result[0]["e"]

    def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> Relationship:
        """Create a relationship between nodes."""
        query = f"""
            MATCH (start), (end)
            WHERE start.id = $start_id
            AND end.id = $end_id
            CREATE (start)-[r:{relationship_type}]->(end)
            SET r = $properties
            RETURN r
        """
        result = self.execute_query(
            query,
            {
                "start_id": start_node_id,
                "end_id": end_node_id,
                "properties": properties or {},
            },
        )
        return result[0]["r"]

    def merge_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        relationship_type: str,
        match_properties: dict[str, Any] | None = None,
        set_properties: dict[str, Any] | None = None,
    ) -> Relationship:
        """Create a relationship between two id-addressed nodes.

        Memgraph versions differ in what property-map matching is supported inside
        MERGE for relationships. To keep ingestion compatible, this method falls
        back to CREATE + SET semantics.
        """
        set_props = set_properties or {}

        edge_id = None
        if match_properties and "edge_id" in match_properties:
            edge_id = match_properties["edge_id"]
        elif "edge_id" in set_props:
            edge_id = set_props["edge_id"]

        if edge_id is not None:
            # Idempotent upsert keyed by edge_id.
            find_query = f"""
                MATCH (start {{id: $start_id}})-[r:{relationship_type} {{edge_id: $edge_id}}]->(end {{id: $end_id}})
                RETURN r
                LIMIT 1
            """
            found = self.execute_query(
                find_query,
                {
                    "start_id": start_node_id,
                    "end_id": end_node_id,
                    "edge_id": edge_id,
                },
            )

            if not found:
                create_query = f"""
                    MATCH (start {{id: $start_id}}), (end {{id: $end_id}})
                    CREATE (start)-[r:{relationship_type}]->(end)
                    SET r.edge_id = $edge_id
                    RETURN r
                """
                self.execute_query(
                    create_query,
                    {
                        "start_id": start_node_id,
                        "end_id": end_node_id,
                        "edge_id": edge_id,
                    },
                )

            set_query = f"""
                MATCH (start {{id: $start_id}})-[r:{relationship_type} {{edge_id: $edge_id}}]->(end {{id: $end_id}})
                SET r += $set_props
                RETURN r
            """
            result = self.execute_query(
                set_query,
                {
                    "start_id": start_node_id,
                    "end_id": end_node_id,
                    "edge_id": edge_id,
                    "set_props": set_props,
                },
            )
            return result[0]["r"]

        # Fallback: not idempotent.
        query = f"""
            MATCH (start {{id: $start_id}}), (end {{id: $end_id}})
            CREATE (start)-[r:{relationship_type}]->(end)
            SET r += $set_props
            RETURN r
        """

        result = self.execute_query(
            query,
            {
                "start_id": start_node_id,
                "end_id": end_node_id,
                "set_props": set_props,
            },
        )
        return result[0]["r"]

    def merge_entity(
        self, label: str, match_property: str, properties: dict[str, Any]
    ) -> Node:
        """Merge an entity node (create if not exists)."""
        query = f"""
            MERGE (e:{label} {{{match_property}: $match_value}})
            SET e += $properties
            RETURN e
        """
        result = self.execute_query(
            query, {"match_value": properties[match_property], "properties": properties}
        )
        return result[0]["e"]

    def find_entity_by_id(
        self, entity_id: str, label: str | None = None
    ) -> dict[str, Any] | None:
        """Find an entity by its ID."""
        label_clause = f":{label}" if label else ""
        query = f"""
            MATCH (e{label_clause} {{id: $id}})
            RETURN e
        """
        results = self.execute_query(query, {"id": entity_id})
        return results[0]["e"] if results else None

    def find_entities_by_type(
        self, entity_type: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Find all entities of a given type."""
        query = f"""
            MATCH (e:{entity_type})
            RETURN e
            LIMIT $limit
        """
        return self.execute_query(query, {"limit": limit})

    def traverse_relationships(
        self,
        entity_id: str,
        relationship_type: str | None = None,
        max_depth: int = 2,
        direction: str = "OUT",
    ) -> list[dict[str, Any]]:
        """Traverse relationships from an entity."""
        rel_clause = f":{relationship_type}" if relationship_type else ""
        direction_symbol = {"OUT": "->", "IN": "<-", "BOTH": "-"}[direction]

        query = f"""
            MATCH (start {{id: $id}})-[r{rel_clause}*1..{max_depth}{direction_symbol}](related)
            RETURN DISTINCT start, r, related
            LIMIT 100
        """
        return self.execute_query(query, {"id": entity_id})

    def close(self):
        """Close connection."""
        if self.client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
