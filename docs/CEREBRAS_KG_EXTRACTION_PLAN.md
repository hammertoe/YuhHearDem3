# Cerebras-Only KG Extraction Plan (gpt-oss-120b)

This document describes the implementation plan to switch the repository's "main" KG extraction flow to Cerebras `gpt-oss-120b` only, using the two-pass + reasoning approach validated in `scripts/compare_kg_models.py`.

Scope is limited to the KG extraction pipeline (windows -> extract -> canonicalize/store). Transcription and embeddings remain unchanged.

## Goals (Hardcoded Defaults)

The updated "main" KG extraction script should use these defaults without CLI knobs:

- Model: `gpt-oss-120b`
- Two pass: `always`
- Reasoning effort: `medium`
- Reasoning format: `hidden`
- Max completion tokens: `16384`
- Embeddings: keep Google embeddings for candidate node retrieval and base KG seeding

The main script path should not import or depend on Gemini/`google.genai` code.

## Current State (as of Feb 2026)

- Main KG extraction entrypoint: `scripts/kg_extract_from_video.py`
  - Seeds base KG with `BaseKGSeeder`
  - Uses `WindowBuilder` to build windows
  - Extracts windows using `lib/knowledge_graph/kg_extractor.py` (`KGExtractor`, Gemini-only)
  - Stores nodes/edges via `KGExtractor.canonicalize_and_store()` (Postgres)

- Gemini KG extractor: `lib/knowledge_graph/kg_extractor.py`
  - Contains:
    - prompt building
    - Gemini API calls
    - parsing helpers (`_parse_edges_from_llm_data` etc.)
    - storage/canonicalization (`canonicalize_and_store`)

- Comparison harness and Cerebras-specific improvements:
  - `scripts/compare_kg_models.py`
  - `lib/knowledge_graph/oss_two_pass.py`
  - Deterministic normalizers:
    - `normalize_utterance_ids_in_data(...)`
    - `normalize_evidence_in_data(...)`
  - OSS prompts:
    - `build_oss_draft_prompt(...)` (recall-oriented)
    - `build_oss_additions_prompt(...)` (pass2 deltas)
    - `merge_oss_additions(...)` (collision-safe merge)

## Key Design Choice

Do **not** rewrite `KGExtractor` in place.

Instead:

1) Create a new Cerebras-only extractor module for the production KG extraction flow.
2) Extract storage/canonicalization into a shared module that has no Gemini imports.
3) Update `scripts/kg_extract_from_video.py` to import only the Cerebras extractor + shared store module.

This avoids accidental Gemini dependencies via module imports.

## Implementation Steps

### Step 1: Create a shared store module (no Gemini imports)

Add a new file:

- `lib/knowledge_graph/kg_store.py`

Move/copy the logic of `KGExtractor.canonicalize_and_store(...)` from `lib/knowledge_graph/kg_extractor.py` into a standalone function:

```python
def canonicalize_and_store(
    *,
    postgres: PostgresClient,
    embedding: GoogleEmbeddingClient,
    results: list[ExtractionResult],
    youtube_video_id: str,
    kg_run_id: str,
    extractor_model: str,
) -> dict[str, Any]:
    ...
```

Constraints:

- Keep behavior identical to current implementation.
- No `google.genai` imports.
- Keep speaker ref normalization, node ID canonicalization, edge ID generation, batching, and embedding updates.

### Step 2: Add a Cerebras-only extractor module

Add a new file:

- `lib/knowledge_graph/oss_kg_extractor.py`

This module implements extraction of a `ConceptWindow` using Cerebras `gpt-oss-120b`.

Hardcode defaults:

- `model = "gpt-oss-120b"`
- `two_pass = always`
- `reasoning_effort = "medium"`
- `reasoning_format = "hidden"`
- `max_completion_tokens = 16384`

Must include:

- Cerebras client creation via `cerebras.cloud.sdk` using `CEREBRAS_API_KEY`.
- Robust call behavior:
  - request `response_format={"type":"json_object"}` when possible
  - retry/fallback when `message.content` is empty
  - allow fallback call without `response_format` if necessary

- Loose JSON parsing for responses:
  - brace-matching extraction of a top-level JSON object if strict parse fails

- Two-pass logic (always):
  - Pass 1 uses OSS recall prompt:
    - `build_oss_draft_prompt(...)`
    - `target_edges = len(window.utterances) + 2`
  - Normalize pass1 output:
    - `normalize_utterance_ids_in_data(data, youtube_video_id=...)`
    - `normalize_evidence_in_data(data, window_text=window.text)`
  - Validate with `validate_kg_llm_data(...)`
  - Pass 2:
    - if 0 violations: run additions-only prompt `build_oss_additions_prompt(...)` and merge via `merge_oss_additions(...)`
    - else: run repair prompt `build_refine_prompt(...)` (deletion allowed)
  - Normalize pass2 output similarly.

- Edge/node conversion:
  - Convert final JSON into extracted node/edge objects consistent with existing storage/canonicalization expectations.
  - You may re-use parsing logic from `KGExtractor._parse_edges_from_llm_data(...)` by copying it into the new module (recommended to avoid importing Gemini code).

### Step 3: Update the main extraction script to be Cerebras-only

Modify:

- `scripts/kg_extract_from_video.py`

Changes:

- Remove Gemini imports and flags:
  - Remove `DEFAULT_GEMINI_MODEL` and `KGExtractor` import.
  - Remove `--model` CLI arg.

- Use the new Cerebras extractor:
  - instantiate `OssKGExtractor(...)` (or similar)

- Keep embeddings:
  - continue constructing `GoogleEmbeddingClient()`
  - keep `BaseKGSeeder` as-is
  - keep `WindowBuilder` for window construction

- Store results via shared store module:
  - call `lib/knowledge_graph/kg_store.py` `canonicalize_and_store(...)`

- Print the hardcoded Cerebras configuration in the script header.

### Step 4: Dependency updates

Update:

- `requirements.txt`

Add:

- `cerebras-cloud-sdk`

Runtime environment:

- Must have `CEREBRAS_API_KEY` set.
- Must have `GOOGLE_API_KEY` set (embeddings remain).

### Step 5: Update tests

Update unit tests that currently use `KGExtractor.canonicalize_and_store`:

- `tests/test_kg_extractor_canonicalize.py`

Change to import and call:

- `lib/knowledge_graph/kg_store.py` `canonicalize_and_store(...)`

This keeps storage logic tested without importing Gemini code.

### Step 6: Verification

After implementation, run a short smoke extraction:

```bash
python scripts/kg_extract_from_video.py --youtube-video-id Syxyah7QIaM --max-windows 3
```

Verify:

- Two-pass runs per window.
- No truncated JSON (max tokens 16384).
- No spurious failures due to bare utterance IDs or evidence not being exact substrings.
- Postgres inserts succeed and stats match expectations.

## Notes and Rationale

- The OSS-specific prompt (`build_oss_draft_prompt`) and the additions-only second pass (`build_oss_additions_prompt` + `merge_oss_additions`) materially improved recall vs using the Gemini prompt unmodified.
- Deterministic normalizers (`normalize_utterance_ids_in_data`, `normalize_evidence_in_data`) prevent avoidable validation failures and keep pass2 in recall mode.
- Keeping embeddings preserves candidate node retrieval quality and avoids reworking the existing vector context approach.
