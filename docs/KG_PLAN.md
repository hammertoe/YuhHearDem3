# KG Plan: LLM-First Canonical KG + Memgraph Projection

## Goals

- Build a high-quality knowledge graph from parliamentary transcripts with:
  - **Gemini-labeled semantic edges** (the main point of the system)
  - **Abstract/concept nodes** (e.g., "Fixed Penalty Regime", "Court Backlog")
  - **Adjacent cross-speaker discourse edges** (`RESPONDS_TO`, `AGREES_WITH`, `DISAGREES_WITH`, `QUESTIONS`)
  - Strong **provenance** for every edge (utterance ids + earliest timestamp + evidence quote)
- Maintain **stable canonical IDs** across runs using **hash-based IDs**
- Use **Postgres as source-of-truth** for canonical KG (nodes, aliases, edges)
- Use **Memgraph as derived traversal store** (projection from Postgres)

## Non-Goals (initial implementation)

- Full offline entity/ontology deduplication/merge workflow (we will store enough info to do it later)
- RDF/Turtle as the primary extraction interface (may export JSON-LD/Turtle later)

## Background / Why the current approach fails

The existing `build_knowledge_graph.py` is NER-first:

- spaCy NER produces mostly named entities (`PERSON`/`ORG`/`GPE`/`DATE`/etc.)
- Gemini is asked to label relationships *only among those entities*
- This misses the main conceptual graph nodes we actually care about:
  - regimes/systems/policies/problems/goals/processes (not reliably named entities)
- Result: very few labeled edges vs the richness visible when prompting an LLM directly.

Therefore: switch to an **LLM-first concept KG extraction** where the model can introduce abstract nodes but is constrained to link to a provided set of known canonical nodes when possible.

---

## High-level architecture

### Data stores

1) **Postgres (canonical KG store)**
- `kg_nodes`: canonical nodes + embeddings
- `kg_aliases`: alias index for fast linking
- `kg_edges`: canonical edges + provenance + confidence + model metadata
- This is the source of truth and is where later offline merging will happen.

2) **Memgraph (projection)**
- Nodes/edges are merged/created from Postgres.
- Used for traversal and product features (graph exploration, API).

### Extraction pipeline (per video / per run)

1) Seed base KG:
- Speakers
- Order paper agenda items (bills / acts)
- Bills table (if available)

2) Build transcript windows from utterances (`sentences` table):
- Concept extraction windows (fixed size + overlap)
- Discourse windows at speaker-change boundaries

3) For each window:
- Retrieve a small candidate set of canonical KG nodes (vector search + alias hits)
- Prompt Gemini to produce a **KG delta**:
  - link to existing canonical node ids when possible
  - create new nodes otherwise
  - emit labeled edges with provenance

4) Canonicalize new nodes (hash ids), upsert into Postgres, write edges into Postgres

5) Sync to Memgraph (MERGE nodes and CREATE/MERGE edges)

---

## Canonical IDs (hash-based)

### Speakers

- ID: `speaker_<speaker_id>` (already stable)

### Other nodes

- ID: `kg_<md5(type:normalized_label)>[:12]` (or similar)
- `normalized_label`: lowercase, trim, collapse whitespace, normalize punctuation

### Edge IDs

- ID: `kge_<md5(source_id|predicate|target_id|youtube_video_id|earliest_seconds|evidence_hash)>[:12]`
- Rationale: allow repeated edges with different evidence/timestamps to coexist if desired, while still allowing dedupe later.

---

## Postgres schema (new tables)

Create a migration file (e.g. `schema/migrations/002_canonical_kg.sql`) with:

### `kg_nodes`

- `id TEXT PRIMARY KEY`
- `label TEXT NOT NULL`
- `type TEXT NOT NULL` (start coarse: `foaf:Person`, `schema:Legislation`, `schema:Organization`, `schema:Place`, `skos:Concept`)
- `aliases TEXT[] DEFAULT '{}'`
- `embedding vector(768)`
- `tsv tsvector`
- `created_at TIMESTAMP DEFAULT NOW()`
- `updated_at TIMESTAMP DEFAULT NOW()`

Indexes:

- vector index on `embedding` (pgvector; choose ivfflat/hnsw depending on availability)
- btree on `(type, id)`
- optional GIN on `tsv`
- optional GIN on `aliases` (later)

### `kg_aliases`

- `alias_norm TEXT PRIMARY KEY`
- `alias_raw TEXT NOT NULL`
- `node_id TEXT NOT NULL REFERENCES kg_nodes(id)`
- `type TEXT NOT NULL`
- `source TEXT NOT NULL` (e.g. `speaker_seed`, `order_paper`, `bill_seed`, `llm`)
- `confidence FLOAT`

Indexes:

- btree on `node_id`
- btree on `type`

### `kg_edges`

- `id TEXT PRIMARY KEY`
- `source_id TEXT NOT NULL`
- `predicate TEXT NOT NULL`
- `target_id TEXT NOT NULL`

Provenance fields:

- `youtube_video_id TEXT NOT NULL`
- `earliest_timestamp_str TEXT`
- `earliest_seconds INTEGER`
- `utterance_ids TEXT[]`
- `evidence TEXT`
- `speaker_ids TEXT[]`
- `confidence FLOAT`
- `extractor_model TEXT`
- `kg_run_id TEXT`
- `created_at TIMESTAMP DEFAULT NOW()`

Indexes:

- btree on `(youtube_video_id, earliest_seconds)`
- btree on `(source_id, predicate, target_id)`

---

## Base KG seeding

### Speakers

From `speakers` table:

- create `speaker_<speaker_id>` nodes with type `foaf:Person`
- aliases: full name, normalized variants, constituency/title if available

### Order paper

From `order_papers` + `order_paper_items`:

- each bill title -> node type `schema:Legislation`
- aliases include common shortened forms (optional at seed time; can be added later)

### Bills table (if present)

From `bills`:

- bill title + bill_number -> `schema:Legislation` nodes
- aliases: bill number, title variants

---

## Windowing strategy

### Input stream

Use `sentences` for a single `youtube_video_id`, ordered by `seconds_since_start`.

Each utterance includes:

- `id` (sentence id)
- `timestamp_str`
- `seconds_since_start`
- `speaker_id`
- `text`

### A) Concept extraction windows

- `window_size=10` utterances
- `stride=6` utterances (overlap=4)
- Filter out ultra-short acknowledgements optionally (e.g. "Yes.", "Good morning.") to reduce noise.

### B) Adjacent discourse windows (adjacent-only requirement)

At every speaker transition A -> B:

- window = last 3 utterances from A + first 3 utterances from B
- discourse extraction only:
  - edges among speakers: `RESPONDS_TO`, `AGREES_WITH`, `DISAGREES_WITH`, `QUESTIONS`
  - must cite utterance ids + evidence quote

---

## Retrieval of candidate nodes for context (stability)

For each concept window:

1) Build query text = concatenation of window utterances
2) Query embedding = `RETRIEVAL_QUERY`
3) Candidate nodes = top-K by vector similarity from `kg_nodes.embedding`:
   - default `K=25`
4) Alias hits:
   - exact match normalized phrases -> `kg_aliases` (optional)
5) Always include:
   - `speaker_<speaker_id>` nodes for speakers present in the window
   - optionally: seeded order paper legislation nodes for that sitting (keep small)

Then provide to the LLM a "Known Nodes" table:

- `id | type | label | aliases[] (optional)`

Keep prompt context bounded:

- target <= 30-60 known nodes per window

---

## LLM extraction contract (delta-based)

### Node types (initial coarse set)

- `foaf:Person`
- `schema:Legislation`
- `schema:Organization`
- `schema:Place`
- `skos:Concept` (default for abstract nodes)

### Predicates (initial allowlist)

Concept edges:

- `AMENDS`
- `GOVERNS`
- `MODERNIZES`
- `AIMS_TO_REDUCE`
- `REQUIRES_APPROVAL`
- `IMPLEMENTED_BY`
- `RESPONSIBLE_FOR`
- `ASSOCIATED_WITH`
- `CAUSES`
- `ADDRESSES`
- `PROPOSES`

Discourse edges (adjacent-only):

- `RESPONDS_TO`
- `AGREES_WITH`
- `DISAGREES_WITH`
- `QUESTIONS`

(These can be iterated later; keep list small initially for quality.)

### Output schema (strict JSON)

For concept windows, Gemini returns:

```json
{
  "nodes_new": [
    {"temp_id": "n1", "type": "skos:Concept", "label": "Fixed penalty regime", "aliases": ["fixed penalties"]}
  ],
  "edges": [
    {
      "source_ref": "speaker_s_mr_ralph_thorne_1",
      "predicate": "PROPOSES",
      "target_ref": "n1",
      "evidence": "I want today ... to offer prescriptions in relation to the Road Traffic Act...",
      "utterance_ids": ["Syxyah7QIaM:2564", "Syxyah7QIaM:2589"],
      "earliest_timestamp": "0:44:09",
      "confidence": 0.72
    }
  ]
}
```

Rules enforced in prompt:

- If the node matches a Known Node, **MUST** use the existing `id` (do not create a new node).
- `predicate` must be from allowlist.
- `evidence` must be a substring quote from the window text.
- `utterance_ids` must refer to provided utterances.
- Output must be JSON only (no markdown).

For discourse windows:

- Return only edges among speakers; no `nodes_new`.

### Constrained decoding / validation

- Use JSON parsing + strict schema validation in code.
- If response is invalid:
  - log raw response
  - optionally retry with a repair prompt (future enhancement)

---

## Canonicalization (online, minimal)

When `nodes_new` arrive:

- canonicalize `label` -> normalized
- generate canonical hash id
- upsert into `kg_nodes`
- insert alias rows into `kg_aliases`

When edges arrive:

- resolve `source_ref` / `target_ref`:
  - if it’s a known id: use it
  - if it’s temp: map via newly created canonical id
- compute edge id
- insert into `kg_edges`

Embeddings:

- embed new node labels (and possibly aliases) using `RETRIEVAL_DOCUMENT` with output dimensionality matching pgvector size (768).

---

## Sync to Memgraph (projection)

Implement a sync path:

- MERGE nodes by `id`
- create edges:
  - relationship type = sanitized predicate (Memgraph-safe)
  - store properties:
    - `edge_id`
    - `evidence`, `utterance_ids`, `earliest_seconds`, `earliest_timestamp`
    - `speaker_ids`, `youtube_video_id`
    - `confidence`, `kg_run_id`, `extractor_model`

Because Memgraph relationship MERGE limitations may exist:

- use pattern:
  - MATCH existing by `(start.id, end.id, edge_id)` then CREATE if missing
  - or simply CREATE and rely on Postgres as truth + periodic rebuild (acceptable early)

---

## Instrumentation / debugging

Per run:

- windows processed
- JSON parse success rate
- edges/window
- new nodes/window
- link rate: % edges that use known node ids vs created nodes
- discourse edges count

Debug flags:

- dump first N prompts and raw responses to a file for inspection
- optionally log per-window stats

---

## CLI / scripts (to implement)

1) `scripts/kg_seed_base.py`
- seeds `kg_nodes` + `kg_aliases` from speakers/order paper/bills

2) `scripts/kg_extract_from_video.py`
- args: `--youtube-video-id`, `--window-size`, `--stride`, `--max-windows`, `--run-id`
- runs concept windows + discourse windows
- writes to Postgres canonical tables
- optionally triggers Memgraph sync

3) `scripts/kg_sync_memgraph.py`
- reads from Postgres and projects into Memgraph
- args: `--youtube-video-id` (optional), `--run-id` (optional)

---

## Testing strategy

Unit tests (no external API):

- mock Gemini client to return valid JSON
- ensure canonicalization:
  - stable hash ids
  - alias upsert
  - edge id determinism
- ensure window building logic is correct
- ensure retrieval query shapes are correct (mock Postgres results)

Integration tests (opt-in):

- run against dockerized Postgres/Memgraph
- seed base KG, run extraction for a small fixture transcript, confirm:
  - `kg_nodes` count increases
  - `kg_edges` inserted with provenance
  - Memgraph nodes/edges appear

---

## Rollout plan

1) Add DB migration + seed base KG
2) Run extraction on a single known video with `--max-windows 30` and inspect:
   - edge density
   - abstract node quality
   - discourse edges presence
3) Tune:
   - candidate K
   - window size/stride
   - predicate allowlist
4) Full run
5) Sync to Memgraph + validate traversal queries

---

## Future enhancements

- Offline merge job (de-dupe/canonicalization improvements)
- JSON-LD export from canonical store
- Predicate mapping to standard vocabularies where appropriate
- Improved candidate retrieval (combine alias + embedding + graph neighborhood)
- Repair/retry on invalid LLM output
