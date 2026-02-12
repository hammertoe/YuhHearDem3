# Search System Architecture

## Overview

YuhHearDem3 includes a hybrid search system combining:
- **Vector Search**: Semantic similarity using pgvector
- **BM25 Full-Text**: Traditional keyword search
- **Graph Traversal**: Multi-hop relationship expansion
- **Hybrid Graph-RAG**: Compact subgraph retrieval with citations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Search Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query                                                      │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────┐                                             │
│  │ Vector Search   │───▶ Top-K semantically similar nodes      │
│  │ (pgvector)     │                                             │
│  └─────────────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  ┌─────────────────┐                                             │
│  │ Graph Expansion │───▶ N-hop traversal from seeds            │
│  │ (PostgreSQL)   │                                             │
│  └─────────────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  ┌─────────────────┐                                             │
│  │ Citation Lookup │───▶ Transcript sentences with provenance  │
│  │                 │                                             │
│  └─────────────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  ┌─────────────────┐                                             │
│  │ Re-ranking      │───▶ Combined score (vector + graph)      │
│  │                 │                                             │
│  └─────────────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  Results with citations + timestamps                            │
└─────────────────────────────────────────────────────────────────┘
```

## Three-Tier Storage

### Tier 1: Paragraphs
- Full paragraphs with speaker attribution
- 768-dimensional embeddings
- Grouped consecutive sentences by same speaker

### Tier 2: Entities
- Named entities (speakers, organizations, legislation)
- Knowledge graph nodes with embeddings
- Aliases for entity linking

### Tier 3: Sentences
- Individual sentences
- Full-text search (BM25)
- Provenance (video, timestamp, speaker)

## Database Queries

### Vector Search (pgvector)
```sql
SELECT id, text, embedding <=> query_vector AS distance
FROM paragraphs
ORDER BY embedding <=> query_vector
LIMIT 20;
```

### Full-Text Search (BM25)
```sql
SELECT id, text,
       ts_rank(tsv, to_tsquery('english', query)) AS rank
FROM sentences
WHERE tsv @@ to_tsquery('english', query)
ORDER BY rank DESC
LIMIT 20;
```

### Graph Traversal
```sql
-- Find related entities within N hops
SELECT DISTINCT target
FROM kg_edges
WHERE source IN (SELECT id FROM kg_nodes WHERE label ILIKE query)
   OR source IN (
     SELECT target FROM kg_edges
     WHERE source IN (SELECT id FROM kg_nodes WHERE label ILIKE query)
   )
LIMIT 50;
```

## Hybrid Graph-RAG

### Process
1. **Seed Retrieval**: Vector search over `kg_nodes`
2. **Graph Expansion**: N-hop traversal from seed nodes
3. **Edge Collection**: Gather all edges in expanded subgraph
4. **Citation Lookup**: Find transcript sentences for each edge
5. **Subgraph Assembly**: Compact representation for LLM

### Output Schema
```json
{
  "query": "...",
  "seeds": [...],           // Seed nodes
  "nodes": [...],           // Expanded nodes
  "edges": [...],          // Edges between nodes
  "citations": [...]        // Transcript evidence
}
```

## API Endpoints

### POST /search
```json
{
  "query": "healthcare reform",
  "limit": 20,
  "alpha": 0.6
}
```

### POST /chat
```json
{
  "thread_id": "uuid",
  "message": "What did they say about healthcare?"
}
```

## Performance

| Operation | Typical Latency |
|-----------|----------------|
| Vector search (Top-10) | <10ms |
| BM25 search (Top-20) | <5ms |
| Graph expansion (2 hops) | <50ms |
| End-to-end search | <100ms |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed_k` | 8 | Number of seed nodes |
| `hops` | 1 | Graph traversal depth |
| `max_edges` | 60 | Maximum edges in subgraph |
| `max_citations` | 12 | Maximum citations |
| `alpha` | 0.6 | Vector/graph balance |
