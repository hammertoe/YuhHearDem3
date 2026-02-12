# YuhHearDem3 - Code Map and Review

## Project Overview

YuhHearDem3 is a parliamentary transcription and knowledge graph system that processes video recordings of parliament sessions, extracts structured information, and enables conversational search over debates.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YuhHearDem3 System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   Video Input   │───▶│  Transcription  │───▶│  Three-Tier Storage    │  │
│  │  (YouTube/GCS)  │    │  (Gemini 2.5)   │    │  (PostgreSQL + pgvector)│ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│           │                      │                        │                   │
│           ▼                      ▼                        ▼                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ Order Papers    │    │ Knowledge Graph │    │     Search API         │  │
│  │ (PDF Parsing)   │───▶│    Extraction   │───▶│  - Hybrid Search       │  │
│  └─────────────────┘    └─────────────────┘    │  - Conversational AI   │  │
│                                                  │  - Graph Traversal     │  │
│                                                  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Code Map

### Entry Points

| File | Lines | Purpose |
|------|-------|---------|
| `transcribe.py` | ~650 | Main video transcription script |
| `api/search_api.py` | ~400 | FastAPI application with chat and search endpoints |

### Library Modules

#### Core Agents (`lib/`)

| File | Lines | Purpose |
|------|-------|---------|
| `chat_agent_v2.py` | 551 | Conversational AI agent with thread management |
| `kg_agent_loop.py` | 699 | KG-powered agent loop with tool calling |
| `kg_hybrid_graph_rag.py` | ~400 | Hybrid Graph-RAG retrieval |
| `advanced_search_features.py` | ~450 | Temporal search, trends, graph queries |

#### Knowledge Graph (`lib/knowledge_graph/`)

| File | Lines | Purpose |
|------|-------|---------|
| `oss_two_pass.py` | 677 | OSS two-pass entity extraction |
| `window_builder.py` | 287 | Window-based processing for transcripts |
| `kg_store.py` | ~350 | KG storage operations |
| `kg_extractor.py` | ~550 | Main KG extraction logic |
| `base_kg_seeder.py` | ~300 | Base KG seeding |
| `model_compare.py` | ~300 | Model comparison utilities |

#### Order Papers (`lib/order_papers/`)

| File | Lines | Purpose |
|------|-------|---------|
| `pdf_parser.py` | 192 | PDF order paper parsing |
| `video_matcher.py` | 344 | Match papers to YouTube videos |
| `ingestor.py` | 95 | Order paper ingestion |
| `parser.py` | 129 | Order paper parsing |
| `models.py` | 34 | Order paper models |
| `role_extract.py` | 27 | Speaker role extraction |

#### Transcripts (`lib/transcripts/`)

| File | Lines | Purpose |
|------|-------|---------|
| `ingestor.py` | 433 | Transcript ingestion |

#### Database (`lib/db/`)

| File | Lines | Purpose |
|------|-------|---------|
| `postgres_client.py` | ~100 | PostgreSQL connection pool |
| `chat_schema.py` | ~150 | Chat schema management |

#### Processors (`lib/processors/`)

| File | Lines | Purpose |
|------|-------|---------|
| `three_tier_transcription.py` | 147 | Three-tier transcript processing |
| `paragraph_splitter.py` | 115 | Paragraph grouping |
| `bill_entity_extractor.py` | 341 | Bill entity extraction |
| `bill_ingestor.py` | 184 | Bill ingestion |

#### Scraping (`lib/scraping/`)

| File | Lines | Purpose |
|------|-------|---------|
| `bill_scraper.py` | 305 | Bill scraping from parliament website |

#### Utilities (`lib/utils/`)

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 85 | Configuration management |
| `roles.py` | ~50 | Role utilities |
| `id_generators.py` | ~100 | ID generation utilities |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `kg_extract_from_video.py` | Extract KG from video |
| `cron_transcription.py` | Automated transcription jobs |
| `migrate_chat_schema.py` | Chat schema migration |
| `clear_kg.py` | Clear KG tables |
| `ingest_order_paper_pdf.py` | Ingest order paper PDFs |
| `ingest_knowledge_graph.py` | Ingest KG data |
| `list_channel_videos.py` | List channel videos |
| `match_order_papers_to_videos.py` | Match papers to videos |
| `backfill_speaker_video_roles.py` | Backfill speaker roles |
| `compare_kg_models.py` | Compare KG models |
| `migrate_transcripts.py` | Migrate transcripts |
| `export_order_paper.py` | Export order papers |
| `deploy.sh` | Deployment script |
| `kg_export_html.py` | Export KG to HTML |
| `kg_sync_memgraph.py` | Sync with Memgraph |
| `kg_seed_base.py` | Seed base KG |

### Tests (`tests/`)

| File | Purpose |
|------|---------|
| `test_chat_agent_v2_unit.py` | Chat agent tests |
| `test_kg_agent_loop_unit.py` | KG agent loop tests |
| `test_kg_hybrid_graph_rag_unit.py` | Graph-RAG tests |
| `test_oss_two_pass.py` | OSS two-pass tests |
| `test_window_builder.py` | Window builder tests |
| `test_trace_helpers_unit.py` | Trace helper tests |
| `test_order_paper_*.py` | Order paper tests |
| `test_bill_*.py` | Bill tests |
| `test_database.py` | Database tests |

## Module Dependencies

```
transcribe.py
├── google.genai (Gemini API)
├── yt_dlp (YouTube metadata)
├── rapidfuzz (fuzzy matching)
├── pydantic (data validation)
├── tenacity (retry logic)
└── lib/order_papers (order paper context)

api/search_api.py
├── fastapi (web framework)
├── lib.chat_agent_v2
├── lib.kg_agent_loop
├── lib.kg_hybrid_graph_rag
└── lib.db.postgres_client

lib/chat_agent_v2.py
├── lib.db.chat_schema
├── lib.kg_agent_loop
└── lib.utils.config

lib/kg_agent_loop.py
├── lib.kg_hybrid_graph_rag
├── google.genai
└── lib.utils.config

lib/knowledge_graph/*.py
├── google.genai (LLM extraction)
├── lib.db.postgres_client
└── tenacity (retry logic)

lib/order_papers/*.py
├── lib.db.postgres_client
└── pdf parsing libraries
```

## Key Design Patterns

### 1. Canonical ID Generation
```python
# Hash-based stable IDs for consistency
kg_<hash(type:label)>[:12]  # Node IDs
kge_<hash(source|predicate|target|video|seconds|evidence)>[:12]  # Edge IDs
```

### 2. Window-Based Processing
```python
# Configurable window size and stride
window_size = 10  # utterances per window
stride = 6  # utterances between windows
# 40% overlap for continuity
```

### 3. Thread-Based Chat
```python
# Persistent conversation threads
chat_threads (table)
├── id: UUID
├── title: str
├── created_at: timestamp
└── updated_at: timestamp

chat_messages (table)
├── id: UUID
├── thread_id: FK
├── role: 'user' | 'assistant'
├── content: text
└── created_at: timestamp
```

### 4. Citation Tracking
```python
# Every answer grounded in evidence
answer → cite_utterance_ids → transcript sentences
```

## Code Review Findings

### Strengths

1. **Clear Architecture**: Separation between transcription, KG extraction, and chat
2. **Modern Python**: Type hints, dataclasses, pydantic models
3. **Error Handling**: Tenacity decorators for retry logic
4. **Database Design**: Well-designed schema with proper indexes
5. **Citation Tracking**: Full provenance for answers
6. **Testing**: Comprehensive unit tests for core functionality

### Areas for Improvement

1. **Large Functions**: Some functions exceed 100 lines
2. **Type Coverage**: Not all functions have type annotations
3. **Documentation**: Some modules lack docstrings
4. **Error Messages**: Could be more descriptive

## Recent Changes

1. **Chat Agent V2**: Complete rewrite with thread-based conversations
2. **KG Extraction**: Added OSS two-pass extraction for improved accuracy
3. **Order Papers**: New PDF parsing and video matching
4. **Tracing**: Comprehensive debug tracing with `CHAT_TRACE`
5. **Follow-Up Questions**: LLM-generated contextual follow-ups

## File Organization

```
YuhHearDem3/
├── api/                    # FastAPI endpoints
├── lib/
│   ├── chat_agent_v2.py   # Chat agent
│   ├── kg_*.py           # Knowledge graph
│   ├── order_papers/      # Order paper processing
│   ├── transcripts/       # Transcript processing
│   ├── db/                # Database clients
│   ├── processors/        # Data processors
│   ├── scraping/          # Web scrapers
│   └── utils/             # Utilities
├── scripts/               # CLI scripts
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── schema/                # Database schema
```

## Summary

The codebase is well-organized with clear separation of concerns. Core functionality includes:

- **Transcription**: Iterative video processing with speaker diarization
- **Knowledge Graph**: LLM-first extraction with canonical IDs
- **Conversational Search**: Thread-based chat with grounded answers
- **Order Papers**: PDF parsing and video matching

Overall quality: **Good** - Solid architecture with comprehensive testing.
