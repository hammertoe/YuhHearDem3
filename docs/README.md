# YuhHearDem3 - Barbados Parliament Search & Knowledge Graph

A comprehensive parliamentary transcription and search system that processes video recordings, extracts knowledge graphs, and enables conversational search over Barbados Parliament debates.

## Features

### Video Transcription
- **Iterative Processing**: Breaks long videos into overlapping segments with context preservation
- **Speaker Consistency**: Maintains speaker IDs across segments using fuzzy name matching
- **Legislation Tracking**: Identifies bills and laws discussed in videos
- **Order Paper Integration**: Uses order papers for context and speaker roles
- **Video Metadata**: Automatically fetches title, duration, and upload date via yt-dlp

### Knowledge Graph Extraction
- **LLM-First Extraction**: Uses Google Gemini to extract entities and relationships in a single pass
- **Window-Based Processing**: Concept windows with configurable size and stride (default: 10 utterances, stride 6)
- **Semantic Relationships**: Captures 15 predicates (11 conceptual + 4 discourse)
- **Canonical IDs**: Hash-based stable node and edge IDs for consistency
- **OSS Two-Pass**: Advanced extraction with improved entity resolution

### Conversational Search
- **Thread-Based Chat**: Persistent conversation threads in PostgreSQL
- **Hybrid Graph-RAG**: Retrieves compact subgraphs with citations
- **Follow-Up Suggestions**: Generates contextual follow-up questions
- **Full Citation Tracing**: Every answer grounded in transcript evidence

### Search System
- **Hybrid Search**: Combines vector similarity, BM25 full-text, and graph traversal
- **Temporal Filters**: Search within date ranges
- **Speaker Filtering**: Filter results by speaker
- **Graph Visualization**: Interactive exploration of entity relationships

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YuhHearDem3 System                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   Video     │───▶│  Transcription  │───▶│  Three-Tier Storage     │  │
│  │  (YouTube)  │    │  (Gemini 2.5)   │    │  (PostgreSQL + pgvector)│  │
│  └─────────────┘    └─────────────────┘    └─────────────────────────┘  │
│         │                   │                         │                    │
│         ▼                   ▼                         ▼                    │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ Order Papers│    │ Knowledge Graph │    │     Search API (FastAPI) │  │
│  │  (PDF)      │───▶│    Extraction   │───▶│     - Hybrid Search      │  │
│  └─────────────┘    └─────────────────┘    │     - Conversational    │  │
│                                             │     - Graph Traversal    │  │
│                                             └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.13+
- PostgreSQL 16+ with pgvector
- Google AI API key

### Installation
```bash
# Clone and install
git clone https://github.com/anomalyco/YuhHearDem3.git
cd YuhHearDem3

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-api-key"
```

### Transcribe a Video
```bash
python transcribe.py --order-file order.txt --segment-minutes 30
```

### Extract Knowledge Graph
```bash
python scripts/kg_extract_from_video.py --youtube-video-id "VIDEO_ID"
```

### Start Chat API
```bash
python -m uvicorn api.search_api:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
YuhHearDem3/
├── api/
│   └── search_api.py              # FastAPI application with all endpoints
├── lib/
│   ├── chat_agent_v2.py           # Conversational AI agent
│   ├── kg_agent_loop.py           # KG-powered agent loop
│   ├── kg_hybrid_graph_rag.py     # Hybrid Graph-RAG retrieval
│   ├── advanced_search_features.py # Temporal search, trends, graph queries
│   ├── knowledge_graph/
│   │   ├── oss_two_pass.py       # OSS two-pass extraction
│   │   ├── window_builder.py      # Window-based processing
│   │   ├── kg_store.py            # KG storage operations
│   │   └── kg_extractor.py        # Main KG extraction
│   ├── order_papers/
│   │   ├── pdf_parser.py          # PDF order paper parsing
│   │   ├── video_matcher.py       # Match papers to videos
│   │   └── ingestor.py            # Order paper ingestion
│   └── transcripts/
│       └── ingestor.py            # Transcript ingestion
├── scripts/
│   ├── kg_extract_from_video.py   # Extract KG from video
│   ├── cron_transcription.py      # Automated transcription
│   ├── migrate_chat_schema.py     # Chat schema migration
│   └── clear_kg.py                # Clear KG tables
├── tests/                          # Unit tests
└── docs/                           # Documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) | Comprehensive implementation guide |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Command quick reference |
| [CHAT_TRACE.md](CHAT_TRACE.md) | Debug tracing documentation |
| [DATE_NORMALIZATION.md](DATE_NORMALIZATION.md) | Date handling |

## Technology Stack

- **Backend**: Python 3.13+, FastAPI, Pydantic
- **Database**: PostgreSQL 16+, pgvector
- **AI**: Google Gemini 2.5 Flash
- **Video**: yt-dlp
- **Search**: Hybrid vector/graph retrieval
- **Testing**: pytest, ruff, mypy

## License

MIT
