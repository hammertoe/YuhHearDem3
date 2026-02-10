# Parliamentary Search System

Hybrid vector/graph search system for parliamentary discussions with three-tier storage architecture.

## Architecture

Three-Tier Storage:
- **TIER 1: Paragraphs** - Have embeddings, group sentences by speaker
- **TIER 2: Entities** - Have embeddings, first-class entities (speakers, bills)
- **TIER 3: Sentences** - No embeddings, context/provenance only

Search Flow:
1. Vector search (entities + paragraphs) → Find relevant content
2. Graph traversal (Memgraph) → Expand relationships
3. Sentence retrieval (PostgreSQL) → Get context with provenance

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.13+
- Google AI API key

### Setup

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   # Edit .env with your GOOGLE_API_KEY
   ```

2. **Start databases:**
   ```bash
   docker-compose up -d
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install python-dotenv==1.0.1
   ```

4. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_md
   ```

5. **Migrate existing data:**
   ```bash
   python scripts/migrate_transcripts.py \
     --transcript-file transcription_output.json \
     --kg-file knowledge_graph.json \
     --video-id Syxyah7QIaM
   ```

6. **Start search API:**
   ```bash
   python api/search_api.py
   ```

## Directory Structure

```
.
├── api/                    # FastAPI application
│   └── search_api.py      # Hybrid search endpoint
├── lib/                     # Core libraries
│   ├── db/                 # Database clients
│   │   ├── postgres_client.py
│   │   └── memgraph_client.py
│   ├── embeddings/          # Embedding services
│   │   └── google_client.py
│   ├── processors/          # Data processing
│   │   └── paragraph_splitter.py
│   └── utils/              # Utilities
│       └── config.py
├── schema/                  # Database schemas
│   └── init.sql          # Three-tier PostgreSQL schema
├── scripts/                 # Ingestion scripts
│   └── migrate_transcripts.py
├── tests/                   # Tests
│   ├── test_database.py
│   ├── test_id_generators.py
│   └── test_paragraph_splitter.py
├── docker-compose.yml        # Database infrastructure
├── requirements.txt          # Python dependencies
├── .env.example            # Configuration template
└── README.md              # This file
```

## Usage

### Docker Commands

```bash
# Start databases
docker-compose up -d

# Stop databases
docker-compose down

# View logs
docker-compose logs -f postgres
docker-compose logs -f memgraph

# Access Memgraph Lab
open http://localhost:7444

# Access PostgreSQL
docker-compose exec postgres psql -U postgres parliament_search
```

### Migration Scripts

**Migrate transcripts:**
```bash
python scripts/migrate_transcripts.py
```

**Options:**
- `--transcript-file`: Path to transcript JSON (default: transcription_output.json)
- `--kg-file`: Path to knowledge graph JSON (default: knowledge_graph.json)
- `--video-id`: YouTube video ID (default: Syxyah7QIaM)

### Search API

**Start API:**
```bash
python api/search_api.py
```

**Search endpoint:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "road traffic safety",
    "limit": 20,
    "alpha": 0.6
  }'
```

**Response format:**
```json
{
  "query": "road traffic safety",
  "results": [
    {
      "sentence_id": "Syxyah7QIaM:2194",
      "text": "Speeding penalties will increase from $200 to $500 to improve road safety.",
      "timestamp": "00:36:34",
      "seconds_since_start": 2194,
      "speaker": {
        "id": "s_hon_santia_bradshaw_1",
        "name": "hon santia bradshaw",
        "position": "Minister of Transport"
      },
      "video": {
        "youtube_id": "Syxyah7QIaM",
        "title": "House Sitting",
        "url": "https://youtube.com/watch?v=Syxyah7QIaM&t=2194"
      },
      "score": 0.92,
      "entities_mentioned": [],
      "match_details": {
        "vector_score": 0.88,
        "graph_score": 0.90
      }
    }
  ],
  "search_metadata": {
    "total_results": 15,
    "entities_matched": 3,
    "paragraphs_matched": 8
  }
}
```

## Testing

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test:**
```bash
pytest tests/test_database.py -v
pytest tests/test_id_generators.py -v
pytest tests/test_paragraph_splitter.py -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=lib --cov=scripts --cov=api --cov-report=html
```

## Database Queries

### Vector Search (PostgreSQL)

```sql
-- Find similar entities
SELECT id, text, type,
       embedding <=> '[...]' as distance
FROM entities
ORDER BY embedding <=> '[...]'
LIMIT 20;

-- Find similar paragraphs
SELECT id, text, speaker_id, start_seconds,
       embedding <=> '[...]' as distance
FROM paragraphs
ORDER BY embedding <=> '[...]'
LIMIT 20;
```

### Full-Text Search (PostgreSQL)

```sql
-- BM25 search on sentences
SELECT id, text, speaker_id, seconds_since_start,
       ts_rank(tsv, to_tsquery('english', 'road traffic')) as rank
FROM sentences
WHERE tsv @@ to_tsquery('english', 'road traffic')
ORDER BY rank DESC
LIMIT 20;
```

### Graph Queries (Memgraph)

```cypher
-- Find related entities (2 hops)
MATCH (e:Entity {id: 'ent_abc123'})-[*1..2]-(related)
RETURN DISTINCT e, related
LIMIT 50;

-- Find speaker connections
MATCH (s1:Speaker)-[:DISCUSSES]->(t:Topic)<-[:DISCUSSES]-(s2:Speaker)
WHERE s1.id = 's_speaker_1'
RETURN s1, s2, t;
```

## Configuration

Environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|----------|
| POSTGRES_HOST | PostgreSQL host | localhost |
| POSTGRES_PORT | PostgreSQL port | 5432 |
| POSTGRES_DATABASE | Database name | parliament_search |
| MEMGRAPH_HOST | Memgraph host | localhost |
| MEMGRAPH_PORT | Memgraph port | 7687 |
| GOOGLE_API_KEY | Google AI API key | Required |
| EMBEDDING_MODEL | Embedding model | models/embedding-001 |
| EMBEDDING_DIMENSIONS | Embedding dimensions | 768 |
| LOG_LEVEL | Logging level | INFO |

## Troubleshooting

**Database connection failed:**
- Check Docker containers are running: `docker-compose ps`
- Check port availability: `lsof -i :5432` and `lsof -i :7687`
- Check environment variables in `.env`

**Embedding generation failed:**
- Verify `GOOGLE_API_KEY` is set correctly
- Check API quota: https://aistudio.google.com/app/apikey

**Migration errors:**
- Verify input JSON files exist and are valid
- Check database tables were created: `docker-compose exec postgres psql -U postgres parliament_search -c "\dt"`

**Search API errors:**
- Check logs for detailed error messages
- Verify embedding dimensions match database schema (768)
- Test with a simple query: curl http://localhost:8000/

## Development

**Code style:**
```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check . --fix

# Type checking
mypy lib/ scripts/ api/
```

**Adding new features:**
1. Add dependencies to `requirements.txt`
2. Create feature module in `lib/`
3. Add tests in `tests/`
4. Update documentation

## Performance

**Storage estimate (200 videos):**
- Paragraphs: ~184 MB
- Sentences: ~12 MB
- Entities: ~1.5 MB
- Graph: ~2 MB
- **Total: ~200 MB**

**Query performance:**
- Vector search: <10ms
- Full-text search: <5ms
- Graph traversal (2 hops): <50ms
- End-to-end search: <100ms

## Next Steps

1. **Phase 2**: Implement bill scraping pipeline
2. **Phase 3**: Real-time transcript ingestion
3. **Phase 5**: RAG-based Q&A
4. **Frontend**: React UI with video player

See detailed implementation plan: `~/.local/share/opencode/plans/hybrid_search_system_plan.md`

## License

MIT
