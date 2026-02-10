# YuhHearDem3 - Code Map and Review

## Project Overview

YuhHearDem3 is a parliamentary transcription and search system that processes video recordings of parliamentary sessions, extracts transcripts with speaker identification, builds knowledge graphs, and provides advanced search capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            YuhHearDem3 System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   Video Input   │───▶│  Transcription  │───▶│  Three-Tier Processing  │ │
│  │  (YouTube/GCS)  │    │  (Gemini API)   │    │   (Paragraphs/Sentences)│ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│           │                      │                       │                  │
│           ▼                      ▼                       ▼                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ Bill Scraping   │    │ Knowledge Graph │    │    Vector/Graph DB      │ │
│  │ (parliament.gov)│───▶│    Builder      │───▶│  (Postgres/Memgraph)    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│                                                     │                       │
│                                                     ▼                       │
│                                           ┌─────────────────────────┐      │
│                                           │   Search API (FastAPI)  │      │
│                                           │  - Hybrid Search        │      │
│                                           │  - Temporal Search      │      │
│                                           │  - Graph Traversal      │      │
│                                           └─────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Code Map

### Entry Points

| File | Lines | Purpose |
|------|-------|---------|
| `transcribe.py` | 938 | Main video transcription script - processes videos in segments |
| `build_knowledge_graph.py` | 686 | Builds knowledge graphs from transcripts using spaCy + Gemini |
| `api/search_api.py` | 580 | FastAPI REST API for hybrid search |

### Library Modules

#### Database Layer (`lib/db/`)

| File | Lines | Purpose |
|------|-------|---------|
| `postgres_client.py` | 83 | PostgreSQL connection pool with pgvector support |
| `memgraph_client.py` | 153 | Memgraph graph database client |

#### Processing Layer (`lib/processors/`)

| File | Lines | Purpose |
|------|-------|---------|
| `three_tier_transcription.py` | 213 | Groups transcripts into paragraphs/sentences |
| `paragraph_splitter.py` | 111 | Groups sentences by speaker into paragraphs |
| `bill_entity_extractor.py` | 222 | Extracts entities from bill text using spaCy |

#### Scraping Layer (`lib/scraping/`)

| File | Lines | Purpose |
|------|-------|---------|
| `bill_scraper.py` | 305 | Scrapes bills from parliament website |

#### Utilities (`lib/utils/`)

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 69 | Environment-based configuration management |

#### Embeddings (`lib/embeddings/`)

| File | Lines | Purpose |
|------|-------|---------|
| `google_client.py` | 76 | Google Gemini embedding client with retry logic |

#### ID Generation & Search (`lib/`)

| File | Lines | Purpose |
|------|-------|---------|
| `id_generators.py` | 84 | ID generators for paragraphs, sentences, speakers, bills |
| `advanced_search_features.py` | 371 | Temporal search, trend analysis, multi-hop queries |

### Scripts (`scripts/`)

| File | Lines | Purpose |
|------|-------|---------|
| `ingest_bills.py` | 254 | Ingests scraped bills into database |
| `migrate_transcripts.py` | ? | Migration utilities (empty in current view) |
| `cron_transcription.py` | ? | Cron job for automated transcription |

### Configuration & Schema

| File | Lines | Purpose |
|------|-------|---------|
| `docker-compose.yml` | 41 | PostgreSQL + pgvector + Memgraph services |
| `schema/init.sql` | 335 | Three-tier database schema with indexes |
| `requirements.txt` | 41 | Python dependencies |

### Tests (`tests/`)

| File | Lines | Purpose |
|------|-------|---------|
| `test_advanced_search.py` | ? | Advanced search feature tests |
| `test_database.py` | ? | Database client tests |
| `test_paragraph_splitter.py` | ? | Paragraph splitting tests |
| `test_id_generators.py` | ? | ID generator tests |
| `test_three_tier_transcription.py` | ? | Three-tier processing tests |
| `test_bill_scraping.py` | ? | Bill scraper tests |
| `test_bill_ingestion.py` | ? | Bill ingestion tests |

## Module Dependency Graph

```
transcribe.py
├── google.genai (Gemini API)
├── yt_dlp (YouTube metadata)
├── rapidfuzz (fuzzy matching)
├── pydantic (data validation)
├── tenacity (retry logic)
└── dotenv (environment)

build_knowledge_graph.py
├── spacy (NER)
├── networkx (graph operations)
├── pyvis (visualization)
├── google.genai (Gemini for relations)
├── tenacity (retry logic)
└── dateparser (date normalization)

api/search_api.py
├── fastapi (web framework)
├── numpy (scoring)
├── lib.db.postgres_client
├── lib.db.memgraph_client
├── lib.embeddings.google_client
├── lib.advanced_search_features
└── lib.utils.config

lib/db/postgres_client.py
├── psycopg
├── psycopg_pool
└── lib.utils.config

lib/db/memgraph_client.py
├── mgclient
└── lib.utils.config

lib/processors/three_tier_transcription.py
├── spacy
├── lib.processors.paragraph_splitter
└── lib.id_generators

lib/processors/paragraph_splitter.py
└── lib.id_generators

lib/scraping/bill_scraper.py
├── requests
├── beautifulsoup4
├── tenacity
└── lib.utils.config

lib/embeddings/google_client.py
├── google.genai
├── tenacity
└── lib.utils.config

lib/advanced_search_features.py
├── lib.db.postgres_client
├── lib.db.memgraph_client
└── lib.embeddings.google_client
```

## Code Review Findings

### Strengths

1. **Good Architecture**: Clear separation between transcription, processing, storage, and search layers
2. **Modern Python**: Uses type hints, dataclasses, pydantic models consistently
3. **Error Handling**: Tenacity decorators for retry logic throughout
4. **Database Design**: Well-designed three-tier schema with appropriate indexes
5. **Configuration Management**: Centralized config using dataclasses and environment variables
6. **Modularity**: Good separation of concerns with processors, db clients, and scrapers

### Issues Found

#### 1. Duplicate NOT_FOUND Constants
- **Location**: `transcribe.py:127` and `transcribe.py:167`
- **Issue**: Same constant defined twice
- **Impact**: Low - second definition shadows the first
- **Fix**: Remove one definition

#### 2. Unused Variables
- **Location**: `transcribe.py:768` - `consecutive_empty_segments` is set but never checked
- **Location**: `build_knowledge_graph.py` - Multiple variables created but not fully utilized
- **Impact**: Low - code clutter
- **Fix**: Remove unused variables or implement the check logic

#### 3. Type Inconsistencies
- **Location**: `build_knowledge_graph.py:9` - Uses `Dict, List, Optional, Tuple, Any` instead of modern syntax
- **Location**: `lib/advanced_search_features.py:5` - Same issue
- **Location**: `lib/scraping/bill_scraper.py:6` - Same issue
- **Impact**: Low - still works but inconsistent with project style
- **Fix**: Use `dict, list, | None` syntax per AGENTS.md guidelines

#### 4. Mutable Default Arguments
- **Location**: `build_knowledge_graph.py:25-33` - Entity dataclass has mutable defaults
- **Impact**: Medium - could cause unexpected behavior if instances share state
- **Fix**: Use `field(default_factory=list)` from dataclasses

#### 5. Inconsistent Error Handling
- **Location**: `transcribe.py:790-806` - Mix of try/except and tenacity retry
- **Issue**: Inner retry loop conflicts with outer loop
- **Impact**: Medium - could cause double retry or unexpected behavior
- **Fix**: Consolidate retry logic

#### 6. Missing Type Hints
- **Location**: `transcribe.py:110-119` - `define_env_vars` missing return type
- **Location**: Multiple functions in test files
- **Impact**: Low - reduces code clarity
- **Fix**: Add type hints

#### 7. Hardcoded Values
- **Location**: `api/search_api.py:442-444` - Hardcoded YouTube video ID
- **Location**: `lib/advanced_search_features.py:100-103` - SQL injection risk in entity_id handling
- **Impact**: Medium - security and maintainability concern
- **Fix**: Use parameterized queries properly

#### 8. SQL Injection Risk
- **Location**: `lib/advanced_search_features.py:111-113`
- **Issue**: Direct string interpolation into SQL
- **Impact**: High - security vulnerability
- **Fix**: Use parameterized queries

#### 9. Resource Management
- **Location**: `build_knowledge_graph.py:118` - Loads spaCy model on every instance
- **Impact**: Medium - slow initialization
- **Fix**: Use singleton pattern or cache model

#### 10. Documentation Gaps
- **Location**: Many functions lack docstrings
- **Location**: Module-level documentation missing in several files
- **Impact**: Medium - harder to maintain
- **Fix**: Add docstrings per Google style

#### 11. Test Coverage Unknown
- **Issue**: Tests exist but coverage not verified
- **Impact**: Unknown - could have untested code paths
- **Fix**: Run tests and generate coverage report

#### 12. Environment Variable Validation
- **Location**: `lib/utils/config.py`
- **Issue**: No validation that required env vars are set
- **Impact**: Medium - runtime errors if misconfigured
- **Fix**: Add validation with clear error messages

#### 13. Import Organization
- **Location**: Several files don't follow AGENTS.md import order
- **Impact**: Low - consistency issue
- **Fix**: Reorganize imports (standard lib → third-party → local)

#### 14. Missing __init__.py Documentation
- **Location**: `lib/db/__init__.py`, `lib/processors/__init__.py`
- **Issue**: Empty init files
- **Impact**: Low - could add module-level docs
- **Fix**: Add docstrings or exports

#### 15. Configuration Validation
- **Location**: `lib/utils/config.py`
- **Issue**: No validation of database connection parameters
- **Impact**: Medium - connection failures at runtime
- **Fix**: Add connection validation or health check

### Security Issues

1. **SQL Injection** in `lib/advanced_search_features.py:111` - entity_id directly interpolated
2. **No Input Sanitization** in API endpoints - should validate query parameters
3. **Hardcoded Credentials** in some docs (need to verify not in code)
4. **No Rate Limiting** on API endpoints

### Performance Issues

1. **N+1 Query Problem** in `api/search_api.py:184-206` - graph_expand_entities queries in loop
2. **SpaCy Model Loading** - loaded multiple times, should be singleton
3. **Embedding Generation** - no batching in some paths
4. **Memory Usage** - loads entire transcript into memory

### Maintainability Issues

1. **Large Functions**: `process_video_iteratively` (200+ lines), `search` endpoint (100+ lines)
2. **Magic Numbers**: Many hardcoded values without constants
3. **Inconsistent Naming**: `speaker_id` vs `id` in different contexts
4. **Missing Abstractions**: Database queries inline in API handlers

## Recommendations

### High Priority
1. Fix SQL injection vulnerability
2. Add input validation to API endpoints
3. Fix mutable default arguments
4. Add proper error handling for database connections

### Medium Priority
1. Refactor large functions into smaller units
2. Add comprehensive docstrings
3. Implement proper logging (remove print statements)
4. Add database connection pooling configuration
5. Cache spaCy model loading

### Low Priority
1. Standardize on modern Python type hints
2. Add type hints to all functions
3. Remove unused variables
4. Consolidate duplicate constants
5. Add module-level documentation

## Testing Status

- Test files exist for major components
- No test runner configuration visible
- No coverage reports generated
- Recommendation: Add pytest configuration and coverage reporting

## Code Quality Metrics

| Metric | Value |
|--------|-------|
| Total Python LOC | ~4,500 |
| Number of modules | 15+ |
| Test files | 7 |
| Documentation files | 18 (in docs/) |
| Average function length | ~25 lines |
| Type hint coverage | ~80% |

## Conclusion

The codebase is well-architected with good separation of concerns. The main issues are:
1. Security vulnerabilities (SQL injection)
2. Some code quality inconsistencies
3. Missing comprehensive error handling
4. Performance optimizations needed for production scale

Overall quality: **Good** (7/10) - Solid foundation with room for improvement in security and maintainability.
