# YuhHearDem3 - Complete Implementation Guide

## Executive Summary

A hybrid vector/graph search and conversational AI system for Barbados Parliament debates. Combines:
- **Video Transcription**: Gemini 2.5 Flash with iterative segment processing
- **Knowledge Graph**: LLM-first extraction with canonical IDs and provenance
- **Conversational Search**: Thread-based chat with Hybrid Graph-RAG
- **Hybrid Search**: Vector similarity + BM25 full-text + graph traversal

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YuhHearDem3 System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   Video Input   │───▶│  Transcription  │───▶│  Three-Tier Storage     │ │
│  │  (YouTube/GCS)  │    │  (Gemini 2.5)   │    │  (PostgreSQL + pgvector)│ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│           │                      │                         │                │
│           ▼                      ▼                         ▼                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ Order Papers    │    │ Knowledge Graph │    │     Search API          │ │
│  │ (PDF Parsing)   │───▶│    Extraction   │───▶│  - Hybrid Search       │ │
│  └─────────────────┘    └─────────────────┘    │  - Conversational AI   │ │
│                                                   │  - Graph Traversal     │ │
│                                                   └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### Video Transcription (`transcribe.py`)

**Features:**
- Iterative segment processing with configurable duration (default 30 min)
- Speaker diarization with fuzzy matching across segments
- Legislation/bill identification
- Order paper integration for context
- Overlap handling for continuity

**Usage:**
```bash
python transcribe.py --order-file order.txt --segment-minutes 30 --start-minutes 0
```

### Knowledge Graph Extraction (`lib/knowledge_graph/`)

**Architecture:**
- **OSS Two-Pass**: Improved entity/relation extraction
- **Window Builder**: Configurable window size/stride (default: 10/6)
- **Canonical IDs**: Hash-based stable identifiers
- **Vector Context**: Top-K similar nodes per window

**Relationship Types:**
- **Conceptual (11)**: AMENDS, GOVERNS, MODERNIZES, AIMS_TO_REDUCE, REQUIRES_APPROVAL, IMPLEMENTED_BY, RESPONSIBLE_FOR, ASSOCIATED_WITH, CAUSES, ADDRESSES, PROPOSES
- **Discourse (4)**: RESPONDS_TO, AGREES_WITH, DISAGREES_WITH, QUESTIONS

**Node Types:**
- `foaf:Person`, `schema:Legislation`, `schema:Organization`, `schema:Place`, `skos:Concept`

**Usage:**
```bash
python scripts/kg_extract_from_video.py --youtube-video-id "VIDEO_ID" --window-size 10 --stride 6
```

### Conversational Search (`lib/chat_agent_v2.py`)

**Components:**
- **KGChatAgentV2**: Main chat agent class
- **KGAgentLoop**: Handles LLM tool calls and Graph-RAG
- **Thread Storage**: PostgreSQL-backed conversation history
- **Citation Engine**: Grounded answers with transcript citations

**Tracing:**
```bash
CHAT_TRACE=1 python -m uvicorn api.search_api:app --reload
```

### Hybrid Search (`lib/kg_hybrid_graph_rag.py`)

**Pipeline:**
1. Vector search over kg_nodes (semantic similarity)
2. Graph expansion (N-hop traversal)
3. Citation retrieval with timestamps
4. Re-ranking by relevance

---

## Data Flow

### Transcription Flow
```
Video URL → yt-dlp metadata → Gemini API → Segment transcription → Speaker normalization → JSON output
```

### Knowledge Graph Flow
```
Transcript → Window Builder (10 utterances, stride 6) → LLM extraction → Canonicalization → KG Store → PostgreSQL
```

### Chat Flow
```
User Query → Embedding → Vector Search → Graph Expansion → LLM Synthesis → Grounded Answer + Citations
```

---

## Database Schema

### Core Tables

**Transcript Tables:**
- `paragraphs`: Paragraphs with embeddings
- `sentences`: Individual sentences with provenance
- `speakers`: Speaker information
- `speaker_video_roles`: Speaker roles per video

**Knowledge Graph Tables:**
- `kg_nodes`: Canonical nodes with embeddings
- `kg_aliases`: Normalized alias index
- `kg_edges`: Edges with provenance (evidence, timestamps, citations)

**Chat Tables:**
- `chat_threads`: Conversation threads
- `chat_messages`: Messages with role and content
- `chat_thread_state`: Persisted state for follow-ups

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/search` | Hybrid search (vector + graph + BM25) |
| POST | `/chat` | Conversational AI with citations |
| GET | `/chat/threads` | List chat threads |
| POST | `/chat/threads` | Create new thread |
| GET | `/chat/threads/{id}` | Get thread messages |
| POST | `/chat/threads/{id}` | Add message to thread |
| GET | `/graph` | Graph data for entity |
| GET | `/speakers` | List all speakers |
| GET | `/speakers/{id}` | Speaker details |

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `transcribe.py` | Main video transcription |
| `scripts/kg_extract_from_video.py` | Extract KG from video |
| `scripts/cron_transcription.py` | Automated transcription jobs |
| `scripts/migrate_chat_schema.py` | Chat schema migration |
| `scripts/clear_kg.py` | Clear KG tables |
| `scripts/ingest_order_paper_pdf.py` | Ingest order paper PDFs |
| `scripts/list_channel_videos.py` | List channel videos |

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google AI Studio API key |
| `CHAT_TRACE` | Enable chat tracing (1/true/on) |
| `ENABLE_THINKING` | Enable model thinking |

### Command-Line Options

**transcribe.py:**
- `--order-file`: Path to order paper file
- `--segment-minutes`: Segment duration (default: 30)
- `--start-minutes`: Start position (default: 0)
- `--max-segments`: Limit segments processed

**kg_extract_from_video.py:**
- `--youtube-video-id`: Video ID to process
- `--window-size`: Utterances per window (default: 10)
- `--stride`: Utterances between windows (default: 6)
- `--max-windows`: Limit windows processed

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_chat_agent_v2_unit.py -v

# Lint
ruff check .

# Type check
mypy lib/
```

---

## Quick Reference

### Essential Commands

```bash
# Transcribe video
python transcribe.py --order-file order.txt

# Extract knowledge graph
python scripts/kg_extract_from_video.py --youtube-video-id "ID"

# Start API
python -m uvicorn api.search_api:app --reload --port 8000

# Run tests
python -m pytest tests/ -v

# Lint
ruff check . --fix
```

### File Locations

| Component | Location |
|-----------|----------|
| Chat API | `api/search_api.py` |
| Chat Agent | `lib/chat_agent_v2.py` |
| KG Extraction | `lib/knowledge_graph/` |
| Order Papers | `lib/order_papers/` |
| Tests | `tests/` |

---

## Dependencies

### Core
- `google-genai>=0.8.0`: Gemini API client
- `fastapi>=0.109.0`: Web framework
- `psycopg[binary,pool]>=3.2.0`: PostgreSQL
- `pydantic>=2.5.0`: Data validation
- `yt-dlp>=2024.0.0`: Video metadata

### Optional
- `rapidfuzz>=3.6.0`: Fuzzy string matching
- `tenacity>=8.2.0`: Retry logic
- `beautifulsoup4>=4.12.0`: HTML parsing

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Command quick reference |
| [CHAT_TRACE.md](CHAT_TRACE.md) | Debug tracing |
| [DATE_NORMALIZATION.md](DATE_NORMALIZATION.md) | Date handling |

---

## License

MIT
