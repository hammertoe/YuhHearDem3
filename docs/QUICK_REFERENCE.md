# YuhHearDem3 - Quick Reference

## Essential Commands

### Transcription
```bash
# Basic transcription
python transcribe.py --order-file order.txt

# With custom segment duration
python transcribe.py --order-file order.txt --segment-minutes 30

# From specific start time
python transcribe.py --order-file order.txt --start-minutes 60

# Limit segments for testing
python transcribe.py --order-file order.txt --max-segments 2
```

### Knowledge Graph Extraction
```bash
# Extract KG from video
python scripts/kg_extract_from_video.py --youtube-video-id "VIDEO_ID"

# With custom window parameters
python scripts/kg_extract_from_video.py --youtube-video-id "VIDEO_ID" --window-size 15 --stride 10

# Limit windows for testing
python scripts/kg_extract_from_video.py --youtube-video-id "VIDEO_ID" --max-windows 5

# Enable debug mode
python scripts/kg_extract_from_video.py --youtube-video-id "VIDEO_ID" --debug
```

### API Server
```bash
# Start chat API
python -m uvicorn api.search_api:app --reload --host 0.0.0.0 --port 8000

# Enable tracing
CHAT_TRACE=1 python -m uvicorn api.search_api:app --reload
```

### Cron Transcription
```bash
# Process watchlist
python scripts/cron_transcription.py --process

# List watchlist
python scripts/cron_transcription.py --list

# Add to watchlist
python scripts/cron_transcription.py --add "VIDEO_ID"

# Remove from watchlist
python scripts/cron_transcription.py --remove "VIDEO_ID"
```

### Database Management
```bash
# Clear KG tables
python scripts/clear_kg.py --yes

# Migrate chat schema
python scripts/migrate_chat_schema.py

# Backfill speaker roles
python scripts/backfill_speaker_video_roles.py
```

### Order Papers
```bash
# Ingest order paper PDF
python scripts/ingest_order_paper_pdf.py --file "order_paper.pdf"

# Match papers to videos
python scripts/match_order_papers_to_videos.py

# Export order paper
python scripts/export_order_paper.py --id "ORDER_ID"
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_chat_agent_v2_unit.py -v
python -m pytest tests/test_kg_agent_loop_unit.py -v

# Lint
ruff check .
ruff check . --fix

# Type check
mypy lib/
```

## API Endpoints

### Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search` | Hybrid search |
| POST | `/chat` | Conversational AI |
| GET | `/chat/threads` | List threads |
| POST | `/chat/threads` | Create thread |
| GET | `/chat/threads/{id}` | Get thread |
| POST | `/chat/threads/{id}` | Send message |
| GET | `/graph` | Graph data |
| GET | `/speakers` | List speakers |
| GET | `/speakers/{id}` | Speaker details |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google AI API key |
| `CHAT_TRACE` | Enable tracing (1/true/on) |
| `ENABLE_THINKING` | Enable model thinking |

## Key Files

| Component | Location |
|-----------|----------|
| Chat Agent | `lib/chat_agent_v2.py` |
| KG Agent Loop | `lib/kg_agent_loop.py` |
| Hybrid Graph-RAG | `lib/kg_hybrid_graph_rag.py` |
| Search API | `api/search_api.py` |
| Main Script | `transcribe.py` |
| KG Extraction | `lib/knowledge_graph/` |
| Order Papers | `lib/order_papers/` |

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview |
| [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) | Full implementation guide |
| [CODE_MAP_AND_REVIEW.md](CODE_MAP_AND_REVIEW.md) | Code structure |
| [CHAT_TRACE.md](CHAT_TRACE.md) | Debug tracing |

## Common Options

### transcribe.py
| Option | Default | Description |
|--------|---------|-------------|
| `--order-file | Path` | Required to order file |
| `--segment-minutes` | 30 | Segment duration |
| `--start-minutes` | 0 | Start position |
| `--max-segments` | None | Limit segments |
| `--output-file` | Varies | Output file path |

### kg_extract_from_video.py
| Option | Default | Description |
|--------|---------|-------------|
| `--youtube-video-id` | Required | Video ID |
| `--window-size` | 10 | Utterances per window |
| `--stride` | 6 | Utterances between windows |
| `--max-windows` | None | Limit windows |
| `--model` | gemini-2.5-flash | Model to use |
| `--debug` | False | Save failed responses |

## Database Tables

### Chat Schema
- `chat_threads` - Conversation threads
- `chat_messages` - Messages with role/content
- `chat_thread_state` - Persisted state

### KG Schema
- `kg_nodes` - Canonical nodes
- `kg_aliases` - Alias index
- `kg_edges` - Edges with provenance

### Transcript Schema
- `paragraphs` - Paragraphs with embeddings
- `sentences` - Sentences with provenance
- `speakers` - Speaker information
- `speaker_video_roles` - Speaker roles per video
