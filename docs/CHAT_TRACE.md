# Chat Trace Logging

## Overview
Chat trace logging provides detailed console output for debugging and understanding of chat agent's behavior during request processing. This is useful for:
- Debugging why LLM made certain tool calls
- Understanding what context is being sent to LLM
- Verifying that citations are being properly extracted and filtered
- Performance tuning by seeing what retrieval operations are happening
- Timing analysis for LLM calls and tool executions

## Enabling Tracing
Set environment variable `CHAT_TRACE` to enable console trace output:
```bash
CHAT_TRACE=1 python -m uvicorn api.search_api:app --reload
# or
CHAT_TRACE=1 python scripts/your_script.py
```

Valid values for `CHAT_TRACE`: `1`, `true`, `on`, `True`, `ON` (case-insensitive)

## Disabling Model Thinking
Set environment variable `ENABLE_THINKING` to enable/disable model thinking:
```bash
# Disable thinking (faster, no thoughts in response)
ENABLE_THINKING=0 python -m uvicorn api.search_api:app --reload

# Enable thinking (default behavior, slower but shows model's reasoning)
ENABLE_THINKING=1 python -m uvicorn api.search_api:app --reload
```

When `ENABLE_THINKING=0`, the agent uses `ThinkingConfig(thinking_budget=0)` which disables thinking output and speeds up responses.

## What Gets Logged

When tracing is enabled, the following information is logged:

1. **Request metadata** (chat agent): thread_id, message IDs, query length
2. **KG agent loop**: user query, history size
3. **Raw contents sent to LLM**: Full, untruncated conversation context with all parts
4. **LLM call duration**: How long the LLM took to respond
5. **Raw LLM response**: Full, untruncated text response from the LLM
6. **Tool execution**: Tool calls made, arguments, duration, and structured results
7. **Final answer**: Length, citations, focus nodes, total duration

> **Note**: Raw content and response logging can produce large output for complex conversations. This is intended for debugging and may flood console in production.

## Trace Output Format
Trace logs use a consistent format with emoji prefixes for easy scanning:
- `🔍 [TRACE {id}] {section}` - Trace section from KG agent loop
- `🔍 [CHAT_TRACE] {section}` - Trace section from chat agent
- `=` separators for visual grouping of sections

Each request gets a unique 8-character trace ID for correlation across components.

## Example Trace Output

```
============================================================
🔍 [CHAT_TRACE] PROCESS MESSAGE START
============================================================
🔍 [CHAT_TRACE] Thread ID
  550e8d4f-e123-4567-89ab-cdef01234567
🔍 [CHAT_TRACE] User Message ID
  a1b2c3d4-e5f6-7890-abcd-ef0123456789
🔍 [CHAT_TRACE] User Query
  What did ministers say about water management? (50 chars)
============================================================

============================================================
🔍 [CHAT_TRACE] AGENT LOOP EXECUTION
============================================================
🔍 [CHAT_TRACE] History Size
  0 messages
============================================================

============================================================
🔍 [TRACE a1b2c3d4] KG AGENT LOOP START
============================================================
🔍 [TRACE a1b2c3d4] User Query
  What did ministers say about water management? (50 chars)
🔍 [TRACE a1b2c3d4] History
  0 messages (last 0 shown if tracing enabled)
============================================================

============================================================
🔍 [TRACE a1b2c3d4] ITERATION 0 - LLM CALL
============================================================
🔍 [TRACE a1b2c3d4] Context Summary
  2 content parts
    [0] system: [{'type': 'text', 'preview': 'You are YuhHearDem...'}]
    [1] user: [{'type': 'text', 'preview': 'What did ministers say...'}]

🔍 [TRACE a1b2c3d4] RAW CONTENTS SENT TO LLM

  [0] Role: system
    [0] Type: text
        Content: You are YuhHearDem, a friendly AI guide to Barbados Parliament debates. Keep a lightly Caribbean (Bajan) tone - warm and plainspoken, not cheesy.

You MUST ground everything in retrieved evidence from the knowledge graph and transcript utterances.

Rules:
- Before answering, call the tool `kg_hybrid_graph_rag` to retrieve a compact subgraph + citations.
- Use ONLY the tool results as your source of truth. Do not invent facts.
- Interpret the tool results: use `edges` + `nodes` to explain relationships in plain language.
- Prefer quoting MPs directly when citations are available; use markdown blockquotes.
- Add visible inline citations in the answer body using markdown links like `[1](#src:utt_123)` immediately after the sentence.
- Use only utterance_ids that appear in the tool citations.
- Use short section headings using markdown like `### The Climate Change Clash`.
- Do NOT include a section called 'Key connections' and do NOT show technical arrow notation.
- Do NOT start your answer with filler like 'Wuhloss,'; start directly with the point.
- Your `answer` field may contain markdown (bullets, bold, blockquotes).
- When you make a claim, include at least one citation by listing its `utterance_id` in `cite_utterance_ids`.
- If evidence is insufficient, say so clearly and ask one precise follow-up.
- Return JSON only matching the response schema.

  [1] Role: user
    [0] Type: text
        Content: What did ministers say about water management?

============================================================

🔍 [TRACE a1b2c3d4] Duration
  847ms

🔍 [TRACE a1b2c3d4] RAW LLM RESPONSE
  Length: 1247 chars
  Content:
```json
{
  "answer": "Based on the retrieved evidence...",
  "cite_utterance_ids": ["utt_123", "utt_456"],
  "focus_node_ids": ["kg_water", "kg_management"]
}
```

============================================================

============================================================
🔍 [TRACE a1b2c3d4] PARSING LLM RESPONSE (iteration 0)
============================================================
🔍 [TRACE a1b2c3d4] Function Calls
  1 call(s): ['kg_hybrid_graph_rag']
============================================================

============================================================
🔍 [TRACE a1b2c3d4] EXECUTING TOOLS (iteration 1)
============================================================
🔍 [TRACE a1b2c3d4] Tool Call
  kg_hybrid_graph_rag(query=water management, hops=1, seed_k=8)
🔍 [TRACE a1b2c3d4] Tool Duration
  324ms
🔍 [TRACE a1b2c3d4] Tool Result
  query='water management', hops=1, seeds_count=8, nodes_count=12, edges_count=24, citations_count=12, nodes_preview=['water', 'funding', 'infrastructure'], citations_preview=['utt_123', 'utt_456', 'utt_789']
🔍 [TRACE a1b2c3d4] Tool Response to LLM
  function_response(name='kg_hybrid_graph_rag') with 1 part(s)
  nodes_count: 12
  edges_count: 24
  citations_count: 12
============================================================

============================================================
🔍 [TRACE a1b2c3d4] ITERATION 1 - LLM CALL
============================================================
🔍 [TRACE a1b2c3d4] Context Summary
  3 content parts
    [0] system: 1 part(s)
        [0] text: You are YuhHearDem, a friendly AI guide...
    [1] user: 1 part(s)
        [0] text: What did ministers say about water...
    [2] user: 1 part(s)
        [0] function_response: function_response(name='kg_hybrid_graph_rag')

🔍 [TRACE a1b2c3d4] RAW CONTENTS SENT TO LLM

  [0] Role: system
    [0] Type: text
        Content: You are YuhHearDem, a friendly AI guide to Barbados Parliament debates...

  [1] Role: user
    [0] Type: text
        Content: What did ministers say about water management?

  [2] Role: user
    [0] Type: function_response
        Name: kg_hybrid_graph_rag
        Response type: dict

============================================================

🔍 [TRACE a1b2c3d4] Duration
  562ms

🔍 [TRACE a1b2c3d4] RAW LLM RESPONSE
  Length: 892 chars
  Content:
```json
{
  "answer": "Ministers discussed water management extensively...",
  "cite_utterance_ids": ["utt_123", "utt_456", "utt_789"],
  "focus_node_ids": ["kg_water", "kg_management"]
}
```

============================================================

============================================================
🔍 [TRACE a1b2c3d4] PARSING LLM RESPONSE (iteration 1)
============================================================
🔍 [TRACE a1b2c3d4] Function Calls
  None - loop complete
============================================================

============================================================
🔍 [TRACE a1b2c3d4] FINAL ANSWER PARSING
============================================================
🔍 [TRACE a1b2c3d4] Final Answer Summary
  length=245 chars, cite_ids=['utt_123', 'utt_456', 'utt_789'], focus_nodes=['kg_water', 'kg_management']
🔍 [TRACE a1b2c3d4] Total Duration
  2.34s
============================================================

============================================================
🔍 [CHAT_TRACE] RESPONSE SUMMARY
============================================================
🔍 [CHAT_TRACE] Assistant Message ID
  b2c3d4e5-f6a7-8901-bcde-f0123456789a
🔍 [CHAT_TRACE] Answer Length
  245 chars
🔍 [CHAT_TRACE] Citation IDs
  3 items
🔍 [CHAT_TRACE] Focus Node IDs
  2 items
🔍 [CHAT_TRACE] Sources Count
  8 items
============================================================
```

Valid values for `CHAT_TRACE`: `1`, `true`, `on`, `True`, `ON` (case-insensitive)

## Trace Output Format
Trace logs use a consistent format with emoji prefixes for easy scanning:
- `🔍 [TRACE {id}] {section}` - Trace section from the KG agent loop
- `🔍 [CHAT_TRACE] {section}` - Trace section from chat agent
- `=` separators for visual grouping of sections

Each request gets a unique 8-character trace ID for correlation across components.

## Example Trace Output

```
============================================================
🔍 [CHAT_TRACE] PROCESS MESSAGE START
============================================================
🔍 [CHAT_TRACE] Thread ID
  550e8d4f-e123-4567-89ab-cdef01234567
🔍 [CHAT_TRACE] User Message ID
  a1b2c3d4-e5f6-7890-abcd-ef0123456789
🔍 [CHAT_TRACE] User Query
  What did ministers say about water management? (50 chars)
============================================================

============================================================
🔍 [CHAT_TRACE] AGENT LOOP EXECUTION
============================================================
🔍 [CHAT_TRACE] History Size
  0 messages
============================================================

============================================================
🔍 [TRACE a1b2c3d4] KG AGENT LOOP START
============================================================
🔍 [TRACE a1b2c3d4] User Query
  What did ministers say about water management? (50 chars)
🔍 [TRACE a1b2c3d4] History
  0 messages (last 0 shown if tracing enabled)
============================================================

============================================================
🔍 [TRACE a1b2c3d4] ITERATION 0 - LLM CALL
============================================================
🔍 [TRACE a1b2c3d4] Context Summary
  2 content parts
    [0] system: [{'type': 'text', 'preview': 'You are YuhHearDem...'}]
    [1] user: [{'type': 'text', 'preview': 'What did ministers say...'}]
============================================================

============================================================
🔍 [TRACE a1b2c3d4] PARSING LLM RESPONSE (iteration 0)
============================================================
🔍 [TRACE a1b2c3d4] Function Calls
  1 call(s): ['kg_hybrid_graph_rag']
============================================================

============================================================
🔍 [TRACE a1b2c3d4] EXECUTING TOOLS (iteration 1)
============================================================
🔍 [TRACE a1b2c3d4] Tool Call
  kg_hybrid_graph_rag(query=water management, hops=1, seed_k=8)
🔍 [TRACE a1b2c3d4] Tool Result Summary
  seeds=8, nodes=12, edges=24, citations=12
🔍 [TRACE a1b2c3d4] Seeds Preview
  ['kg_water_management (skos:Concept)', 'kg_funding (skos:Concept)', 'kg_infrastructure (skos:Concept)']...

🔍 [TRACE a1b2c3d4] ACTUAL KG DATA SENT TO LLM

  Nodes (first 5 of 12):
    [0] kg_water (skos:Concept): Water
    [1] kg_funding (skos:Concept): Funding
    [2] kg_infrastructure (skos:Concept): Infrastructure
    [3] kg_ministry_of_health (foaf:Person): Ministry of Health
    [4] kg_sanitation (skos:Concept): Sanitation

  Edges (first 5 of 24):
    [0] kge_abc123: Water --[DISCUSSES]--> Funding
    [1] kge_def456: Ministry of Health --[RESPONSIBLE_FOR]--> Water
    [2] kge_ghi789: Funding --[ALLOCATED_FOR]--> Infrastructure
    [3] kge_jkl012: Infrastructure --[REQUIRES]--> Sanitation
    [4] kge_mno345: Water --[AFFECTS]--> Sanitation

  Citations (first 5 of 12):
    [0] utt_123: The Honourable John Smith said 'We need to invest in water...'
    [1] utt_456: Minister Jones said 'Funding for water management...'
    [2] utt_789: Dr. Williams said 'Infrastructure improvements are critic...'
    [3] utt_abc: Senator Green said 'Sanitation access remains a challenge...'
    [4] utt_def: The Honourable Johnson said 'Our budget allocation...'

============================================================

============================================================
🔍 [TRACE a1b2c3d4] ITERATION 1 - LLM CALL
============================================================
🔍 [TRACE a1b2c3d4] Context Summary
  3 content parts
============================================================

============================================================
🔍 [TRACE a1b2c3d4] PARSING LLM RESPONSE (iteration 1)
============================================================
🔍 [TRACE a1b2c3d4] Function Calls
  None - loop complete
============================================================

============================================================
🔍 [TRACE a1b2c3d4] FINAL ANSWER PARSING
============================================================
🔍 [TRACE a1b2c3d4] Final Answer Summary
  length=245 chars, cite_ids=['utt_123', 'utt_456', 'utt_789'], focus_nodes=['kg_water', 'kg_management']
============================================================

============================================================
🔍 [CHAT_TRACE] RESPONSE SUMMARY
============================================================
🔍 [CHAT_TRACE] Assistant Message ID
  b2c3d4e5-f6a7-8901-bcde-f0123456789a
🔍 [CHAT_TRACE] Answer Length
  245 chars
🔍 [CHAT_TRACE] Citation IDs
  3 items
🔍 [CHAT_TRACE] Focus Node IDs
  2 items
🔍 [CHAT_TRACE] Sources Count
  8 items
============================================================
```

## Trace Sections Explained

### 1. Raw Contents Sent to LLM
Shows the complete, untruncated conversation context being sent to the LLM:
- Each message with its role (system, user, model)
- For each message, all parts with full content
- For text parts: complete text content
- For function_call parts: name and full args object
- For function_response parts: name and response data type

### 2. Raw LLM Response
Shows the complete, untruncated text response from the LLM:
- Total character count
- Full response text including any markdown, JSON, or free-form content
- Useful for debugging LLM reasoning, tool calling patterns, or response format issues

### 3. Actual KG Data Sent to LLM (NEW)
Shows the complete knowledge graph data being sent back to the LLM:
- **Nodes**: First 5 of total nodes with ID, type, and label
  - Format: `{node_id} ({node_type}): {label}`
- **Edges**: First 5 of total edges with ID, source, predicate, target
  - Format: `{edge_id}: {source_label} --[{predicate}]--> {target_label}`
- **Citations**: First 5 of total citations with utterance_id, speaker, text preview
  - Format: `{utterance_id}: {speaker} said '{text_preview}...'`

This section shows the actual evidence the LLM has access to when synthesizing its answer.

### 4. Tool Result Summary
Shows the complete KG search result returned from `kg_hybrid_graph_rag`:
- `query` - The search query (truncated to 50 chars)
- `hops` - Number of graph hops expanded
- `seeds_count` - Number of seed nodes retrieved
- `nodes_count` - Number of nodes in subgraph
- `edges_count` - Number of edges in subgraph
- `citations_count` - Number of transcript citations
- `nodes_preview` - First 3 node labels (if ≤5 nodes)
- `citations_preview` - First 3 utterance IDs (if ≤5 citations)

### 5. Final LLM Call Context
Shows the complete conversation context being sent to the LLM for final answer synthesis:
- Message count and roles (system, user, function_response)
- For each message: number of parts
- First 2 parts per message with type and text preview
- Duration of the final LLM call

### 6. Duration Tracking
Timings are shown at multiple points:
- Each LLM call duration (e.g., "847ms", "2.34s")
- Each tool execution duration (e.g., "324ms")
- Total loop duration from start to finish

## Implementation Details

### Trace Helper Functions (`lib/kg_agent_loop.py`)
- `_should_trace()` - Check if tracing is enabled
- `_start_timer()` / `_end_timer()` - High-precision timing using `time.perf_counter()`
- `_format_duration()` - Format duration as ms (sub-second) or seconds (≥1s) with 2 decimals
- `_truncate_text()` - Safely truncate text for previews
- `_format_node()` - Format a single KG node (id, type, label)
- `_format_edge()` - Format a single KG edge (id, source, predicate, target)
- `_format_citation()` - Format a single citation (utterance_id, speaker, text preview)
- `_format_content_part_summary()` - Format a content part for logging (truncated preview)
- `_serialize_content_part()` - Serialize a content part to dict for raw logging (full content)
- `_format_contents_summary()` - Format full contents list for logging (previews)
- `_serialize_contents()` - Serialize full contents list for raw logging
- `_format_tool_result_summary()` - Format KG search result for readable output
- `_trace_print()` - Print a trace message with consistent formatting
- `_trace_section_start()` / `_trace_section_end()` - Print section headers/footers

### Chat Agent Tracing (`lib/chat_agent_v2.py`)
- Prints request start information (thread_id, message_id, query)
- Prints history size sent to the loop
- Prints response summary (answer length, citations, focus nodes, sources)

### KG Agent Loop Tracing (`lib/kg_agent_loop.py`)
- Prints user query and history size
- Prints context summary before each LLM call (truncated previews)
- Prints raw contents being sent to LLM (full, untruncated)
- Prints function calls extracted from LLM response
- Prints tool execution summary (for kg_hybrid_graph_rag: seeds, nodes, edges, citations)
- Prints raw LLM response text (full, untruncated)
- Prints final answer summary (length, cite_ids, focus_nodes)

## Tests
Unit tests for trace helpers are in `tests/test_trace_helpers_unit.py`:
- `test_truncate_text_should_shorten_long_text`
- `test_truncate_text_should_not_modify_short_text`
- `test_truncate_text_should_handle_empty_text`
- `test_format_contents_summary_should_handle_empty_list`
- `test_format_contents_summary_should_handle_none_parts`
