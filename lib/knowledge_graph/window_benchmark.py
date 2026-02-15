"""Window benchmark aggregation and reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a window benchmark run."""

    window_size: int
    stride: int
    model: str = "gpt-oss-120b"
    reasoning_effort: str = "medium"
    two_pass: bool = True


@dataclass
class WindowBenchmarkMetrics:
    """Metrics for a single window extraction."""

    window_index: int
    window_size: int
    stride: int
    elapsed_s: float
    parse_success: bool
    nodes_count: int
    edges_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    pass2_triggered: bool
    error: str | None = None

    pass2_elapsed_s: float | None = None
    pass2_tokens: int | None = None


def compute_percentile(values: list[float], p: float) -> float:
    """Compute percentile of sorted values."""
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


@dataclass
class AggregatedBenchmarkMetrics:
    """Aggregated metrics across all windows for a config."""

    config: BenchmarkConfig

    total_windows: int = 0
    successful_windows: int = 0
    failed_windows: int = 0

    total_elapsed_s: float = 0.0
    avg_elapsed_s: float = 0.0
    p50_elapsed_s: float = 0.0
    p95_elapsed_s: float = 0.0
    max_elapsed_s: float = 0.0

    total_tokens: int = 0
    tokens_per_window: float = 0.0

    total_nodes: int = 0
    total_edges: int = 0

    edges_per_1k_tokens: float = 0.0

    pass2_triggered_count: int = 0
    pass2_trigger_rate: float = 0.0
    avg_pass2_elapsed_s: float = 0.0

    success_rate: float = 0.0


def aggregate_metrics(
    metrics: list[WindowBenchmarkMetrics],
    config: BenchmarkConfig,
) -> AggregatedBenchmarkMetrics:
    """Aggregate window metrics into summary statistics."""
    if not metrics:
        return AggregatedBenchmarkMetrics(config=config)

    successes = [m for m in metrics if m.parse_success]
    failures = [m for m in metrics if not m.parse_success]

    elapsed_times = [m.elapsed_s for m in metrics if m.elapsed_s > 0]

    total_tokens = sum(m.total_tokens for m in metrics)
    total_nodes = sum(m.nodes_count for m in successes)
    total_edges = sum(m.edges_count for m in successes)

    pass2_triggered = [m for m in metrics if m.pass2_triggered]
    pass2_elapsed = [m.pass2_elapsed_s for m in pass2_triggered if m.pass2_elapsed_s is not None]

    result = AggregatedBenchmarkMetrics(
        config=config,
        total_windows=len(metrics),
        successful_windows=len(successes),
        failed_windows=len(failures),
        total_elapsed_s=sum(elapsed_times),
        total_tokens=total_tokens,
        total_nodes=total_nodes,
        total_edges=total_edges,
        pass2_triggered_count=len(pass2_triggered),
    )

    if successes:
        result.success_rate = len(successes) / len(metrics)
        result.edges_per_1k_tokens = (
            (total_edges / total_tokens * 1000) if total_tokens > 0 else 0.0
        )

    if elapsed_times:
        result.avg_elapsed_s = sum(elapsed_times) / len(elapsed_times)
        result.p50_elapsed_s = compute_percentile(elapsed_times, 50)
        result.p95_elapsed_s = compute_percentile(elapsed_times, 95)
        result.max_elapsed_s = max(elapsed_times)

    if metrics:
        result.tokens_per_window = total_tokens / len(metrics)

    if len(pass2_triggered) > 0:
        result.pass2_trigger_rate = len(pass2_triggered) / len(metrics)
        if pass2_elapsed:
            result.avg_pass2_elapsed_s = sum(pass2_elapsed) / len(pass2_elapsed)

    return result


def compute_comparison_report(
    results: list[AggregatedBenchmarkMetrics],
) -> dict[str, Any]:
    """Compute comparison report across benchmark configs."""
    report: dict[str, Any] = {
        "configs": [],
        "comparison": {},
    }

    for r in results:
        key = f"w{r.config.window_size}_s{r.config.stride}"
        report["configs"].append(key)
        report["comparison"][key] = {
            "window_size": r.config.window_size,
            "stride": r.config.stride,
            "total_windows": r.total_windows,
            "successful_windows": r.successful_windows,
            "failed_windows": r.failed_windows,
            "success_rate": r.success_rate,
            "total_elapsed_s": r.total_elapsed_s,
            "avg_elapsed_s": r.avg_elapsed_s,
            "p50_elapsed_s": r.p50_elapsed_s,
            "p95_elapsed_s": r.p95_elapsed_s,
            "max_elapsed_s": r.max_elapsed_s,
            "total_tokens": r.total_tokens,
            "tokens_per_window": r.tokens_per_window,
            "total_nodes": r.total_nodes,
            "total_edges": r.total_edges,
            "edges_per_1k_tokens": r.edges_per_1k_tokens,
            "pass2_trigger_rate": r.pass2_trigger_rate,
        }

    return report
