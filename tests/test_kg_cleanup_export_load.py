"""Tests for KG cleanup export/load module."""

import tempfile
from pathlib import Path

from lib.knowledge_graph.cleanup.export_load import (
    export_to_csv,
    export_metrics,
    export_all_artifacts,
    load_nodes_from_csv,
    load_edges_from_csv,
    load_aliases_from_csv,
    load_merge_map_from_csv,
    verify_loaded_data,
)


def test_export_to_csv():
    """Test CSV export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_path = tmpdir_path / "test.csv"

        data = [{"col1": "value1", "col2": "value2"}, {"col1": "value3", "col2": "value4"}]
        export_to_csv(data, output_path, ["col1", "col2"])

        assert output_path.exists()
        content = output_path.read_text()
        assert "value1" in content
        assert "value2" in content


def test_export_to_csv_ignores_extra_fields():
    """Test CSV export ignores unexpected row keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_path = tmpdir_path / "test_extra.csv"

        data = [{"col1": "value1", "col2": "value2", "extra": "ignored"}]
        export_to_csv(data, output_path, ["col1", "col2"])

        assert output_path.exists()
        content = output_path.read_text()
        assert "value1" in content
        assert "value2" in content
        assert "ignored" not in content


def test_export_metrics():
    """Test metrics export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        metrics_path = tmpdir_path / "metrics.json"

        metrics = {
            "nodes_before": 100,
            "nodes_after": 80,
            "edges_before": 200,
            "edges_after": 150,
            "merge_count": 20,
        }
        export_metrics(metrics, metrics_path)

        assert metrics_path.exists()
        content = metrics_path.read_text()
        assert '"nodes_before": 100' in content
        assert '"merge_count": 20' in content


def test_export_all_artifacts():
    """Test export of all cleanup artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        nodes = [
            {
                "id": "node1",
                "label": "Test",
                "type": "foaf:Person",
                "aliases": ["test"],
                "pagerank_score": 0.12,
                "merge_cluster_id": "node1",
                "merged_from_count": 2,
            }
        ]
        edges = [{"id": "edge1", "source_id": "node1", "target_id": "node2"}]
        aliases = [{"alias_norm": "test", "node_id": "node1"}]
        merge_map = {"node1": "node1"}
        edge_drops = [("edge1", "invalid")]
        metrics = {"test": "metrics"}

        export_all_artifacts(nodes, edges, aliases, merge_map, edge_drops, metrics, tmpdir_path)

        assert (tmpdir_path / "kg_nodes_clean.csv").exists()
        assert (tmpdir_path / "kg_edges_clean.csv").exists()
        assert (tmpdir_path / "kg_aliases_clean.csv").exists()
        assert (tmpdir_path / "node_merge_map.csv").exists()
        assert (tmpdir_path / "edge_drop_log.csv").exists()
        assert (tmpdir_path / "metrics_after.json").exists()

        nodes_csv = (tmpdir_path / "kg_nodes_clean.csv").read_text()
        assert "merge_cluster_id" in nodes_csv
        assert "merged_from_count" in nodes_csv


def test_load_nodes_from_csv():
    """Test loading nodes from CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "nodes.csv"

        data = [{"id": "node1", "label": "Test", "type": "foaf:Person"}]
        export_to_csv(data, csv_path, ["id", "label", "type"])

        loaded = load_nodes_from_csv(csv_path)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "node1"
        assert loaded[0]["label"] == "Test"


def test_load_edges_from_csv():
    """Test loading edges from CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "edges.csv"

        data = [{"id": "edge1", "source_id": "node1", "target_id": "node2", "predicate": "GOVERNS"}]
        export_to_csv(data, csv_path, ["id", "source_id", "target_id", "predicate"])

        loaded = load_edges_from_csv(csv_path)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "edge1"
        assert loaded[0]["predicate"] == "GOVERNS"


def test_load_aliases_from_csv():
    """Test loading aliases from CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "aliases.csv"

        data = [{"alias_norm": "test", "node_id": "node1"}]
        export_to_csv(data, csv_path, ["alias_norm", "node_id"])

        loaded = load_aliases_from_csv(csv_path)
        assert len(loaded) == 1
        assert loaded[0]["alias_norm"] == "test"


def test_load_merge_map_from_csv():
    """Test loading merge map from CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "merge_map.csv"

        data = [{"old_node_id": "node1", "new_node_id": "node2"}]
        export_to_csv(data, csv_path, ["old_node_id", "new_node_id"])

        loaded = load_merge_map_from_csv(csv_path)
        assert len(loaded) == 1
        assert loaded["node1"] == "node2"


def test_verify_loaded_data():
    """Test data verification."""
    nodes = [
        {"id": "node1", "label": "Test", "type": "foaf:Person"},
        {"id": "node2", "label": "Test2", "type": "schema:Organization"},
    ]
    edges = [{"id": "edge1", "source_id": "node1", "target_id": "node2", "predicate": "GOVERNS"}]
    metrics = {"nodes_after": 2, "edges_after": 1}

    result = verify_loaded_data(nodes, edges, metrics)
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["metrics"]["nodes_count"] == 2


def test_verify_loaded_data_detects_count_mismatch():
    """Test verification fails when expected counts do not match."""
    nodes = [{"id": "node1", "label": "Test", "type": "foaf:Person"}]
    edges = []
    metrics = {"nodes_after": 2, "edges_after": 1}

    result = verify_loaded_data(nodes, edges, metrics)

    assert result["valid"] is False
    assert any("Node count mismatch" in error for error in result["errors"])
    assert any("Edge count mismatch" in error for error in result["errors"])


def test_verify_loaded_data_detects_discourse_type_violation():
    """Test verification fails when discourse edge is not Person->Person."""
    nodes = [
        {"id": "node1", "label": "A", "type": "foaf:Person"},
        {"id": "node2", "label": "B", "type": "schema:Legislation"},
    ]
    edges = [
        {
            "id": "edge1",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "RESPONDS_TO",
        }
    ]
    metrics = {"nodes_after": 2, "edges_after": 1}

    result = verify_loaded_data(nodes, edges, metrics)

    assert result["valid"] is False
    assert any("Discourse edge type violations" in error for error in result["errors"])


def test_verify_loaded_data_detects_non_allowed_type_and_predicate():
    """Test verification flags non-allowlisted node types and predicates."""
    nodes = [{"id": "node1", "label": "X", "type": "schema:Person"}]
    edges = [
        {
            "id": "edge1",
            "source_id": "node1",
            "target_id": "node1",
            "predicate": "AGREE_WITH",
        }
    ]
    metrics = {"nodes_after": 1, "edges_after": 1}

    result = verify_loaded_data(nodes, edges, metrics)

    assert result["valid"] is False
    assert any("non-allowed node types" in error for error in result["errors"])
    assert any("non-allowed predicates" in error for error in result["errors"])
