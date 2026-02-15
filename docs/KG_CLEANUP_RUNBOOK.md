# KG Cleanup Pass - Operator Runbook

## Overview

This runbook provides step-by-step instructions for running the KG cleanup pass pipeline.

## Prerequisites

- PostgreSQL database with canonical KG tables
- Python 3.13+ with dependencies from requirements.txt
- Sufficient disk space for artifacts (~100MB per run)

## Quick Start

```bash
# Profile current KG state
python scripts/kg_cleanup_pass.py --mode profile --out-dir artifacts/kg_cleanup/$(date +%Y%m%d_%H%M%S)

# Dry-run cleanup (no DB mutation)
python scripts/kg_cleanup_pass.py --mode dry-run --aggressive --same-type-only --out-dir artifacts/kg_cleanup/$(date +%Y%m%d_%H%M%S)

# Export cleaned artifacts
python scripts/kg_cleanup_pass.py --mode export --aggressive --same-type-only --out-dir artifacts/kg_cleanup/$(date +%Y%m%d_%H%M%S)

# Verify exported data
python scripts/kg_cleanup_pass.py --mode verify --out-dir artifacts/kg_cleanup/$(date +%Y%m%d_%H%M%S)
```

## Execution Modes

### 1. Profile Mode

Generates baseline metrics JSON with current KG state:

```json
{
  "profile_timestamp": "2026-02-12T12:00:00",
  "total_nodes": 33840,
  "total_edges": 81260,
  "type_counts": {
    "foaf:Person": 15000,
    "schema:Legislation": 5000
  },
  "predicate_counts": {
    "GOVERNS": 8000,
    "PROPOSES": 6000
  },
  "non_allowed_types": ["schema:Person"],
  "non_allowed_predicates": ["AGREE_WITH"],
  "non_allowed_type_count": 1,
  "non_allowed_predicate_count": 1
}
```

**Output:** `profile_before.json`

### 2. Dry-Run Mode

Runs full cleanup pipeline without database mutations:

- Applies type/predicate remapping
- Generates same-type candidate pairs
- Computes merge clusters
- Computes PageRank and edge ranking scores
- Cleans and collapses edges
- Exports all artifacts to output directory

**Output:** All CSV files + `metrics_after.json`

### 3. Export Mode

Same as dry-run but intended for production export.

**Output:** All CSV files + `metrics_after.json`

### 4. Verify Mode

Validates exported CSV files for consistency:

- Checks all edge endpoints exist in nodes
- Verifies metrics match actual counts

**Output:** Console verification results

## Artifacts

| Artifact | Description |
|----------|-------------|
| `kg_nodes_clean.csv` | Cleaned nodes with ranking fields |
| `kg_edges_clean.csv` | Cleaned edges with support and ranking fields |
| `kg_aliases_clean.csv` | Cleaned aliases |
| `node_merge_map.csv` | Node ID merge mappings |
| `edge_drop_log.csv` | Dropped edges with reasons |
| `metrics_after.json` | Post-cleanup metrics |

## Options

- `--aggressive`: Enable aggressive cleanup mode (type/predicate remapping)
- `--same-type-only`: Only merge same-type nodes (default: enabled)
- `--no-same-type-only`: Allow cross-type deduplication
- `--min-duplicate-reduction-pct`: Minimum duplicate reduction percentage required for verification (default: 3.0)
- `--out-dir <path>`: Output directory for artifacts

## Acceptance Criteria

Run completes successfully when:

- ✅ All output files created in output directory
- ✅ No errors in verification mode
- ✅ Metrics show >= 3% duplicate reduction (default; configurable via --min-duplicate-reduction-pct)
- ✅ All nodes have allowed types
- ✅ All predicates are allowed
- ✅ Discourse edges are Person->Person only

## Troubleshooting

### Memory Issues

If encountering OOM errors during ranking:

1. Process by node type separately
2. Reduce batch size in candidate generation
3. Use `--same-type-only` to limit scope

### Verification Failures

If verification shows errors:

1. Check `edge_drop_log.csv` for dropped edges
2. Verify merge map in `node_merge_map.csv`
3. Check `metrics_after.json` for expected vs actual counts

### Slow Performance

For large graphs (>50K nodes):

1. Profile mode first to identify bottlenecks
2. Consider increasing PostgreSQL work_mem
3. Run on machine with >16GB RAM
