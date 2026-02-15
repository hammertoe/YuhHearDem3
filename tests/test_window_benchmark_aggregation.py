"""Tests for window benchmark aggregation logic."""

from __future__ import annotations


from lib.knowledge_graph.window_benchmark import (
    BenchmarkConfig,
    WindowBenchmarkMetrics,
    aggregate_metrics,
    compute_percentile,
)


def test_compute_percentile_empty() -> None:
    assert compute_percentile([], 50) == 0.0


def test_compute_percentile_single() -> None:
    assert compute_percentile([5.0], 50) == 5.0


def test_compute_percentile_even() -> None:
    result = compute_percentile([1.0, 2.0, 3.0, 4.0], 50)
    assert result == 2.5


def test_compute_percentile_odd() -> None:
    result = compute_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
    assert result == 3.0


def test_compute_percentile_p95() -> None:
    result = compute_percentile(list(range(1, 101)), 95)
    assert 94 <= result <= 96


def test_aggregate_metrics_empty() -> None:
    result = aggregate_metrics([], BenchmarkConfig(window_size=10, stride=6))
    assert result.total_windows == 0
    assert result.total_tokens == 0
    assert result.total_edges == 0


def test_aggregate_metrics_single_window() -> None:
    metrics = WindowBenchmarkMetrics(
        window_index=0,
        window_size=10,
        stride=6,
        elapsed_s=1.5,
        parse_success=True,
        nodes_count=3,
        edges_count=5,
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        pass2_triggered=False,
    )
    result = aggregate_metrics([metrics], BenchmarkConfig(window_size=10, stride=6))
    assert result.total_windows == 1
    assert result.successful_windows == 1
    assert result.failed_windows == 0
    assert result.total_elapsed_s == 1.5
    assert result.total_tokens == 300
    assert result.total_nodes == 3
    assert result.total_edges == 5
    assert result.tokens_per_window == 300.0
    assert result.edges_per_1k_tokens == (5 / 300) * 1000


def test_aggregate_metrics_with_pass2() -> None:
    metrics = WindowBenchmarkMetrics(
        window_index=0,
        window_size=20,
        stride=12,
        elapsed_s=3.0,
        parse_success=True,
        nodes_count=4,
        edges_count=8,
        prompt_tokens=150,
        completion_tokens=400,
        total_tokens=550,
        pass2_triggered=True,
        pass2_tokens=250,
    )
    result = aggregate_metrics([metrics], BenchmarkConfig(window_size=20, stride=12))
    assert result.pass2_triggered_count == 1
    assert result.pass2_trigger_rate == 1.0


def test_aggregate_metrics_multiple_windows() -> None:
    metrics_list = [
        WindowBenchmarkMetrics(
            window_index=i,
            window_size=10,
            stride=6,
            elapsed_s=1.0 + i,
            parse_success=True,
            nodes_count=2 + i,
            edges_count=4 + i,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            pass2_triggered=False,
        )
        for i in range(3)
    ]
    result = aggregate_metrics(metrics_list, BenchmarkConfig(window_size=10, stride=6))
    assert result.total_windows == 3
    assert result.successful_windows == 3
    assert result.failed_windows == 0
    assert result.total_elapsed_s == 6.0
    assert result.total_tokens == 900
    assert result.total_nodes == 9
    assert result.total_edges == 15
    assert result.tokens_per_window == 300.0
    assert result.avg_elapsed_s == 2.0
    assert result.p50_elapsed_s == 2.0
    assert 2.8 <= result.p95_elapsed_s <= 3.0


def test_aggregate_metrics_with_failures() -> None:
    metrics_list = [
        WindowBenchmarkMetrics(
            window_index=0,
            window_size=10,
            stride=6,
            elapsed_s=1.0,
            parse_success=True,
            nodes_count=2,
            edges_count=4,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            pass2_triggered=False,
        ),
        WindowBenchmarkMetrics(
            window_index=1,
            window_size=10,
            stride=6,
            elapsed_s=0.5,
            parse_success=False,
            nodes_count=0,
            edges_count=0,
            prompt_tokens=100,
            completion_tokens=0,
            total_tokens=100,
            pass2_triggered=False,
            error="parse error",
        ),
    ]
    result = aggregate_metrics(metrics_list, BenchmarkConfig(window_size=10, stride=6))
    assert result.total_windows == 2
    assert result.successful_windows == 1
    assert result.failed_windows == 1
    assert result.total_tokens == 400
    assert result.success_rate == 0.5
