# AI Agent Instructions for YuhHearDem3

This file contains build commands, testing procedures, and coding style guidelines for agentic coding.

## Documentation Table of Contents

The `docs/` directory contains comprehensive project documentation:

| Document | Purpose | Lines |
|----------|---------|-------|
| [README.md](docs/README.md) | Main project documentation - transcription, knowledge graphs, usage | 425 |
| [COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md) | Comprehensive 5-phase implementation guide with architecture | 1121 |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Quick reference card for commands and API endpoints | 376 |
| [README_SEARCH_SYSTEM.md](docs/README_SEARCH_SYSTEM.md) | System architecture and database queries | 328 |
| [CODE_MAP_AND_REVIEW.md](docs/CODE_MAP_AND_REVIEW.md) | Complete code map, architecture diagram, and code review | ~400 |
| [DATE_NORMALIZATION.md](docs/DATE_NORMALIZATION.md) | Technical documentation for date normalization | 136 |
| [KG_SUMMARY.md](docs/KG_SUMMARY.md) | Knowledge graph statistics and sample data | 71 |

**Quick Links:**
- Start here: [README.md](docs/README.md) for transcription and knowledge graphs
- For development: [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) for commands
- For architecture: [COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md) for full system overview
- For code understanding: [CODE_MAP_AND_REVIEW.md](docs/CODE_MAP_AND_REVIEW.md)

## Build, Lint, and Type Check Commands

### Linting (ruff)
```bash
# Run linter
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Check specific file
ruff check transcribe.py

# Show detailed output with line numbers
ruff check . --output-format=full
```

### Type Checking (mypy)
```bash
# Run type checker
mypy *.py

# Check specific file
mypy transcribe.py

# More verbose output
mypy *.py --show-error-codes
```

### Running Scripts
```bash
# Transcribe video (main script)
python transcribe.py --order-file order.txt --segment-minutes 30 --start-minutes 0

# Build knowledge graph
python build_knowledge_graph.py --input-file transcription_output.json --output-json knowledge_graph.json --output-html knowledge_graph.html --batch-size 10

# With custom output files
python transcribe.py --order-file order.txt --output-file my_transcription.json
```

### Testing

**Run all tests:**
```bash
python -m pytest tests/ -v
```

**Run specific test file:**
```bash
python -m pytest tests/test_database.py -v
python -m pytest tests/test_advanced_search.py -v
```

**Quick manual testing:**
1. Run the script with `--max-segments 2` for quick iteration
2. Verify JSON output structure matches expected format
3. Check for runtime errors with `python transcribe.py --help`

## Code Style Guidelines

### Imports
- **Order**: Standard library → third-party → local (if any)
- **No blank line** between imports from the same package
- Use `from collections.abc import Callable` for types
- Import from `typing` module for backward compatibility if needed

Example:
```python
import argparse
import json
from datetime import datetime

import pydantic
import tenacity
from google import genai
from google.genai.types import GenerateContentConfig, Part
```

### Type Hints
- **Use modern Python 3.13+ syntax**: `dict[str, str]` instead of `Dict[str, str]`
- **Always annotate function parameters and return types**
- Use `| None` for optional types instead of `Optional[str]`
- Use `list[T]` instead of `List[T]`
- Use `dataclass` for simple data objects
- Use `pydantic.BaseModel` for validated data structures

Example:
```python
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class VideoSegment:
    start: timedelta
    end: timedelta

def get_video_duration(video: Video) -> timedelta | None:
    """Fetch video duration."""
    # implementation
```

### Naming Conventions
- **Functions and variables**: `snake_case`
- **Classes**: `CamelCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Module-level constants**: Define at top after imports
- **Enum classes**: `CamelCase` with `UPPER_CASE` values

Example:
```python
class Model(enum.Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"

DEFAULT_CONFIG = GenerateContentConfig(temperature=0.0)

def format_known_speakers(known_speakers: dict) -> str:
    """Format known speakers for prompt context."""
    pass
```

### Docstrings
- **Keep it simple**: One-line triple-quoted string on first line of function
- Use Google-style docstrings only for complex functions needing more explanation
- No docstring for trivial functions (obvious from name and signature)

Example:
```python
def get_video_metadata_ytdlp(video: Video) -> VideoMetadataInfo:
    """Fetch video metadata using yt-dlp."""
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(video.value, download=False)
    return VideoMetadataInfo(...)
```

### Error Handling
- **Use try/except** for operations that may fail (API calls, file I/O)
- **Print errors** to console with context (e.g., `print(f"Error: {e}")`)
- **Return empty/default values** on failure when appropriate (e.g., `return []`)
- **Raise ValueError** with descriptive message for invalid user input
- Use **tenacity** for retrying transient errors (API rate limits)

Example:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_entities(self, transcript_entry: dict) -> list[tuple]:
    """Extract entities from transcript."""
    try:
        # implementation
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []
```

### Formatting and Line Length
- **Target line length**: 88-100 characters (flexible for URLs, long strings)
- ruff will auto-fix most issues with `ruff check . --fix`
- Use **f-strings** for string formatting (not `.format()` or `%`)
- Use **match/case** instead of if/elif chains for complex conditionals

### File Organization
- **Entry point**: Always use `if __name__ == "__main__":` pattern
- **argparse** setup in main() or at module level if simple
- **Data models**: Define after imports, before functions
- **Helper functions**: Organize logically, typically grouped by purpose

### API Integration
- Use **tenacity** decorators for retry logic on API calls
- Set sensible **timeout** and **max retries** (3-7 attempts)
- Handle **ClientError** specifically for API errors
- Log/retry transient errors (429 rate limits, 503 service unavailable)

### Console Output
- **Progress indicators**: Use `--batch-size` for periodic updates
- **Error prefixes**: Use `❌` for errors, `⚠️` for warnings, `✅` for success
- Use **print()** for all output (no logging module currently)
- Section headers with `=` or `─` separators for readability

### Pattern Matching
Use modern Python 3.10+ match/case for cleaner conditional logic:

```python
match err.code:
    case 400 if "try again" in err.message:
        retry = True
    case 429:
        retry = True
    case _:
        retry = False
```

## Project-Specific Patterns

### Video Transcription (transcribe.py)
- **Enum for video sources**: `Video` class with YouTube URLs
- **Enum for models**: `Model` with Gemini variants
- **Time formats**: Use `timedelta` for all time calculations
- **Speaker IDs**: Format `s_<name>_<number>` (normalized, lowercase, underscore-separated)
- **Legislation IDs**: Format `L_BILLNUMBER_<number>`

### Knowledge Graph (kg_extractor.py)
- **LLM-first approach**: Single pass extraction using Gemini, no NER pre-filtering
- **Window-based extraction**: Concept windows (10 utterances, stride 6) with 40% overlap
- **Node IDs**: Hash-based stable IDs: `kg_<hash(type:label)>[:12]`
- **Edge IDs**: Hash-based: `kge_<hash(source|predicate|target|video|seconds|evidence)>[:12]`
- **Predicates**: 15 predicates (11 conceptual + 4 discourse) extracted in single pass
- **Timestamp accuracy**: Edges use timestamps from specific utterances referenced by each edge
- **Vector context**: Top-K vector search provides relevant known nodes to LLM per window
- **Node types**: `foaf:Person`, `schema:Legislation`, `schema:Organization`, `schema:Place`, `skos:Concept`

### Clear KG Tables (scripts/clear_kg.py)
- Use `python scripts/clear_kg.py --yes` to clear all KG tables for fresh extraction
- Clears: `kg_edges`, `kg_aliases`, `kg_nodes` (in that order due to FK constraints)

## Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install all dependencies
pip install -r requirements.txt

# Set API key (Google AI Studio)
export GOOGLE_API_KEY="your-api-key"
```

## Key Dependencies
- **google-genai**: Gemini API client
- **pydantic**: Data validation
- **tenacity**: Retry logic
- **yt-dlp**: Video metadata
- **rapidfuzz**: Fuzzy string matching
- **psycopg**: PostgreSQL client
- **pgvector**: Vector similarity search
