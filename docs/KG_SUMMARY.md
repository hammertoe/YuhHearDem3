# Knowledge Graph Summary

## Overview

The knowledge graph stores canonical entities and relationships extracted from parliamentary transcripts using LLM-first extraction with Gemini.

## Node Types

| Type | Namespace | Description |
|------|-----------|-------------|
| Person | `foaf:Person` | Individual MPs and speakers |
| Organization | `schema:Organization` | Government bodies, committees |
| Legislation | `schema:Legislation` | Bills, acts, resolutions |
| Place | `schema:Place` | Geographic locations |
| Concept | `skos:Concept` | Policies, topics, abstract concepts |

## Relationship Types

### Conceptual (11 predicates)
| Predicate | Description |
|-----------|-------------|
| `AMENDS` | Legislation amends another |
| `GOVERNS` | Entity governs/regulates |
| `MODERNIZES` | Updates a system/process |
| `AIMS_TO_REDUCE` | Goal-oriented |
| `REQUIRES_APPROVAL` | Needs approval |
| `IMPLEMENTED_BY` | Implementation entity |
| `RESPONSIBLE_FOR` | Responsibility |
| `ASSOCIATED_WITH` | General association |
| `CAUSES` | Causal relationship |
| `ADDRESSES` | Addresses topic/problem |
| `PROPOSES` | Proposes legislation/idea |

### Discourse (4 predicates)
| Predicate | Description |
|-----------|-------------|
| `RESPONDS_TO` | Speaker response |
| `AGREES_WITH` | Agreement |
| `DISAGREES_WITH` | Disagreement |
| `QUESTIONS` | Question asked |

## ID Generation

### Node IDs
```
kg_<hash(type:label)>[:12]
```
Example: `kg_abc123def456`

### Edge IDs
```
kge_<hash(source|predicate|target|video|seconds|evidence)>[:12]
```
Example: `kge_xyz789abc012`

## Provenance

Each edge includes:
- `youtube_video_id`: Source video
- `earliest_timestamp_str`: When mentioned
- `earliest_seconds`: Timestamp in seconds
- `utterance_ids`: Transcript segments
- `evidence`: Quote from transcript
- `speaker_ids`: Speakers involved
- `confidence`: Extraction confidence
- `kg_run_id`: Extraction run identifier

## Sample Data

### Nodes
```json
{
  "id": "kg_abc123def456",
  "label": "Water Management",
  "type": "skos:Concept",
  "aliases": ["water policy", "water governance"],
  "embedding": [...]
}
```

### Edges
```json
{
  "id": "kge_xyz789abc012",
  "source_id": "kg_abc123def456",
  "predicate": "PROPOSES",
  "target_id": "kg_def789ghi012",
  "youtube_video_id": "Syxyah7QIaM",
  "earliest_timestamp_str": "00:36:00",
  "earliest_seconds": 2160,
  "utterance_ids": ["utt_123", "utt_456"],
  "evidence": "The Minister proposed a new water management framework...",
  "speaker_ids": ["s_hon_santia_bradshaw_1"],
  "confidence": 0.95
}
```

## Storage

### Tables
- `kg_nodes`: Canonical nodes with embeddings
- `kg_aliases`: Normalized alias index for entity linking
- `kg_edges`: Edges with full provenance

### Indexes
- `kg_nodes.label`: Fast lookup by label
- `kg_nodes.type`: Filter by node type
- `kg_aliases.alias_norm`: Fuzzy matching
- `kg_edges.source_id` / `kg_edges.target_id`: Graph traversal
- `kg_edges.youtube_video_id`: Filter by video

## Extraction Process

1. **Window Building**: Group utterances (default: 10, stride 6)
2. **LLM Extraction**: Extract entities and relationships
3. **Canonicalization**: Generate stable IDs
4. **Vector Context**: Retrieve similar known nodes
5. **Merging**: Combine with existing KG
6. **Storage**: Persist to PostgreSQL

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 10 | Utterances per window |
| `stride` | 6 | Advance between windows |
| `top_k` | 25 | Similar nodes for context |
| `max_edges` | 60 | Max edges per extraction |

## Performance

| Metric | Typical Value |
|--------|---------------|
| Windows/hour | ~100 |
| Nodes/window | 5-15 |
| Edges/window | 10-30 |
| Extraction latency | <2s/window |
