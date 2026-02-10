# Parliamentary Search System - Quick Reference Card

## 🎯 What Was Built

A hybrid vector/graph search system for parliamentary discussions with:
- ✅ Three-tier storage (paragraphs → entities → sentences)
- ✅ Hybrid search (embeddings + BM25 + graph traversal)
- ✅ Bill scraping & ingestion pipeline
- ✅ Real-time transcript ingestion
- ✅ Advanced search (temporal, trends, multi-hop)
- ✅ Modern React + TypeScript frontend
- ✅ Interactive graph visualization
- ✅ Video player with timestamp navigation
- ✅ Speaker profiles with statistics

---

## 📊 Project Stats

| Metric | Value |
|---------|-------|
| **Total Phases** | 5 (All Complete ✅) |
| **Python Files** | 30+ |
| **TypeScript Files** | 11 |
| **Total Lines of Code** | ~7,000+ |
| **Test Cases** | 60+ |
| **API Endpoints** | 7 |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Python
pip install -r requirements.txt

# Node.js
cd frontend
npm install
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY
```

### 3. Start Databases
```bash
docker-compose up -d
```

### 4. Initialize Database
```bash
psql -U postgres -d parliament_search -f schema/init.sql
```

### 5. Start Backend
```bash
uvicorn api.search_api:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start Frontend
```bash
cd frontend
npm run dev
```

### 7. Open in Browser
```
http://localhost:3000
```

---

## 🌐 API Endpoints

### Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| POST | `/search` | Hybrid search |
| POST | `/search/temporal` | Temporal search with filters |
| GET | `/search/trends` | Trend analysis |
| GET | `/graph` | Graph data |
| GET | `/speakers` | List all speakers |
| GET | `/speakers/{id}` | Get speaker details |

### API Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📁 Key Files

### Backend
- `api/search_api.py` - Main FastAPI application
- `schema/init.sql` - Database schema
- `lib/db/postgres_client.py` - PostgreSQL client
- `lib/db/memgraph_client.py` - Memgraph client
- `lib/embeddings/google_client.py` - Google embeddings
- `lib/advanced_search_features.py` - Advanced search
- `scripts/migrate_transcripts.py` - Data migration
- `scripts/ingest_bills.py` - Bill ingestion
- `scripts/cron_transcription.py` - Watchlist manager

### Frontend
- `frontend/src/App.tsx` - Root component
- `frontend/src/pages/SearchPage.tsx` - Search interface
- `frontend/src/pages/SpeakerPage.tsx` - Speaker profiles
- `frontend/src/pages/GraphPage.tsx` - Graph visualization
- `frontend/src/components/VideoPlayer.tsx` - YouTube player
- `frontend/src/services/api.ts` - API client
- `frontend/src/types/index.ts` - TypeScript types

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_database.py -v

# Lint
ruff check . --fix

# Type check
mypy *.py
```

---

## 🛠️ Common Commands

```bash
# Start everything (run in separate terminals)
make db-up
uvicorn api.search_api:app --reload --port 8000
cd frontend && npm run dev

# Stop databases
make db-down

# Run tests
make test

# Lint
make lint

# Migrate data
make migrate

# Scrape bills
make scrape-bills

# Process transcripts (from watchlist)
python scripts/cron_transcription.py --process

# List watchlist
python scripts/cron_transcription.py --list

# Add to watchlist
python scripts/cron_transcription.py --add "VIDEO_ID"
```

---

## 📱 Frontend Features

### Search Page
- Free-text search
- Date range filters
- Speaker filter
- Entity type filter
- Results with scores
- Click to play video

### Video Player
- Embedded YouTube player
- Automatic timestamp jump
- Play/pause controls
- Close button

### Graph Visualization
- Interactive graph
- Drag/zoom/pan
- Color-coded nodes
- Edge labels
- Tooltips
- Legend

### Speaker Profiles
- Grid view of all speakers
- Speaker details on click
- Statistics cards
- Recent contributions

---

## 🔧 Technology Stack

### Backend
- Python 3.13+
- FastAPI 0.109.0
- PostgreSQL 16 + pg_vector
- Memgraph
- Google AI embedding-001 (768-dim)
- spaCy 3.7.5
- Uvicorn 0.30.1

### Frontend
- React 18.3.1
- TypeScript 5.4.3
- Vite 5.2.8
- Axios 1.6.7
- vis-network 9.1.6
- react-player 2.16.0
- Tailwind CSS 3.4.3

### DevOps
- Docker Compose
- pytest
- ruff
- mypy

---

## 📖 Documentation

- `COMPLETE_GUIDE.md` - This comprehensive guide
- `README.md` - Project overview
- `README_SEARCH_SYSTEM.md` - System architecture
- `AGENTS.md` - AI agent guidelines
- `PHASE1_COMPLETE.md` - Phase 1 details
- `PHASE2_COMPLETE.md` - Phase 2 details
- `PHASE3_COMPLETE.md` - Phase 3 details
- `PHASE4_QUICK_REFERENCE.md` - Phase 4 details
- `PHASE5_COMPLETE_FINAL.md` - Phase 5 details
- `frontend/README.md` - Frontend guide

---

## 🎓 Learning Resources

### Architecture
1. Three-tier storage model
2. Hybrid search pipeline (5-phase)
3. Vector similarity search
4. Graph traversal (2-hop)

### Code Patterns
1. Connection pooling
2. Retry logic with tenacity
3. Type-safe APIs (Pydantic, TypeScript)
4. Component composition (React)
5. State management (React hooks)

### Best Practices
1. Environment variables for secrets
2. Parameterized queries (SQL injection prevention)
3. Error handling with try-catch
4. Linting and type checking
5. Test-driven development

---

## ✅ What Works

### ✅ Backend
- FastAPI server running on port 8000
- PostgreSQL + pg_vector working
- Memgraph graph database working
- Google embeddings generated
- All 7 API endpoints functional
- CORS enabled for frontend

### ✅ Frontend
- React app running on port 3000
- All 3 pages rendering correctly
- API communication working
- Video player functional
- Graph visualization interactive
- Speaker profiles displaying

### ✅ Integration
- Frontend can access all backend endpoints
- Vite proxy routing correctly
- CORS headers present
- Type-safe API calls

---

## 🐛 Known Limitations

1. Entity type filter not implemented in temporal search SQL
2. No pagination for speakers list
3. Generic error messages
4. No authentication/authorization
5. No rate limiting
6. No caching
7. Frontend has no error boundaries
8. No pagination for search results

---

## 🚀 Next Steps (Optional)

### Frontend Enhancements
- [ ] Add error boundaries
- [ ] Skeleton loading screens
- [ ] Better error messages
- [ ] Pagination for results
- [ ] Search history
- [ ] Saved searches
- [ ] Bill detail pages
- [ ] Entity detail pages

### Backend Enhancements
- [ ] Implement entity type filtering
- [ ] Add pagination
- [ ] Better error messages
- [ ] Add authentication (JWT)
- [ ] Add rate limiting
- [ ] Add caching (Redis)
- [ ] Add metrics/logging
- [ ] Auto-generate API docs

### DevOps
- [ ] Dockerize backend
- [ ] Dockerize frontend
- [ ] CI/CD pipeline
- [ ] Multi-environment config
- [ ] Health check endpoints

---

## 📞 Support

### Getting Help

**Documentation**:
- Start with `COMPLETE_GUIDE.md` for comprehensive guide
- Check phase-specific files for detailed information

**Testing**:
- Run tests to verify installation: `python -m pytest tests/ -v`
- Check backend logs for errors
- Check browser console for frontend errors

**Common Issues**:
- See "Troubleshooting" section in `COMPLETE_GUIDE.md`

---

## 🎉 Summary

**Status**: All 5 Phases Complete ✅

**Built**: A fully functional hybrid vector/graph search system for parliamentary discussions with modern React frontend.

**Ready to use**: Just follow the "Quick Start" instructions above!

**Lines of Code**: ~7,000+ lines across 40+ files

**Technology**: Python 3.13+, React 18, TypeScript 5, PostgreSQL 16, Memgraph, FastAPI, Vite

---

**Enjoy the Parliamentary Search System!** 🙏
