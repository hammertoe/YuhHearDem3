# Video Transcription with Gemini

Iterative video transcription with speaker diarization, legislation tracking, and long-video support. Includes knowledge graph extraction from transcripts.

## Features

- **Iterative Processing**: Breaks long videos into overlapping segments
- **Speaker Consistency**: Maintains speaker IDs across segments using fuzzy name matching
- **Legislation Tracking**: Identifies bills and laws discussed in videos
- **Video Metadata**: Automatically fetches title, duration, and upload date via yt-dlp
- **Deduplication**: Removes overlapping content from adjacent segments
- **Context Preservation**: Passes last 5 sentences to next segment for continuity

## Requirements

```bash
pip install google-genai pydantic tenacity yt-dlp rapidfuzz
```

## Configuration

Set one of the following environment variables:

### Option A - Google AI Studio API
```bash
export GOOGLE_API_KEY="your-api-key"
```

Get an API key from: https://aistudio.google.com/app/apikey

### Option B - Vertex AI (Google Cloud)
```bash
export GOOGLE_GENAI_USE_VERTEXAI="True"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="your-location"
```

## Usage

### Basic Usage
```bash
python test.py --order-file order.txt
```

### With Custom Segment Duration
```bash
python test.py --order-file order.txt --segment-minutes 20
```

### With Custom Start Time
```bash
python test.py --order-file order.txt --start-minutes 60
```

### Limited Number of Segments
```bash
python test.py --order-file order.txt --max-segments 5
```

### Custom Output File
```bash
python test.py --order-file order.txt --output-file my_transcription.json
```

## CLI Arguments

| Argument | Type | Default | Description |
|----------|--------|----------|-------------|
| `--start-minutes` | int | 0 | Start time in minutes |
| `--order-file` | str | None | Path to order file for additional context |
| `--segment-minutes` | int | 30 | Duration of each segment in minutes |
| `--overlap-minutes` | int | 1 | Overlap between segments in minutes |
| `--max-segments` | int | None | Maximum number of segments to process |
| `--output-file` | str | transcription_output.json | Output JSON file path |

## Output Format

### Console Output

The script prints three sections to console:

1. **TRANSCRIPTION**: Full transcript with speaker IDs
2. **SPEAKER REFERENCES**: Mapping of voice IDs to speaker IDs and details
3. **LEGISLATION REFERENCES**: List of bills/laws mentioned
4. **VIDEO METADATA**: Title, date, and duration

Example:
```
[0:05:15] S_John_Smith_ABC_1: Welcome everyone to today's session...
[0:06:30] S_Jane_Doe_XYZ_1: Let's discuss the new legislation...

SPEAKER REFERENCES (3 speakers)

S_John_Smith_ABC_1:
  Voice ID: 1
  Name: John Smith
  Position: Director

LEGISLATION REFERENCES (2 items)

L_HR1234_1:
  Name: House Resolution 1234
  Description: Environmental protection bill
  Source: audio

VIDEO METADATA
Title: Committee Meeting January 2025
Upload Date: 20250115
Duration: 2:30:00
```

### JSON Output

Saved to file specified by `--output-file` (default: `transcription_output.json`).

Structure:
```json
{
  "video_metadata": {
    "title": "Video Title",
    "upload_date": "20250115",
    "duration": "2:30:00"
  },
  "transcripts": [
    {
      "start": "0:05:15",
      "text": "Welcome everyone...",
      "voice_id": 1,
      "speaker_id": "S_John_Smith_ABC_1"
    }
  ],
  "speakers": [
    {
      "speaker_id": "S_John_Smith_ABC_1",
      "voice_id": 1,
      "name": "John Smith",
      "position": "Director",
      "role_in_video": "Host"
    }
  ],
  "legislation": [
    {
      "id": "L_HR1234_1",
      "name": "House Resolution 1234",
      "description": "Environmental protection bill",
      "source": "audio"
    }
  ],
  "generated_at": "2025-01-15T10:30:00"
}
```

## Speaker ID Format

Speaker IDs are generated using simple format: `s_<name>_<number>`

Example: `s_j_o_bradshaw_1` (LLM normalizes "Santia J O'Bradshaw")

- `s_` = Speaker prefix (lowercase)
- `<name>` = Speaker's name as normalized by LLM (lowercase, no special characters, underscores for spaces)
- `<number>` = Sequential number (1, 2, 3...) for duplicate normalized names

The LLM is responsible for:
- Normalizing speaker names consistently across all segments
- Removing punctuation (apostrophes, periods, etc.)
- Converting spaces to underscores
- Using lowercase

Examples (as processed by LLM):
- "John Smith" -> `s_john_smith`
- "Santia J O'Bradshaw" -> `s_j_o_bradshaw`
- "Robert L Johnson Jr." -> `s_robert_l_johnson_jr`

**Fuzzy Matching**: Speakers with similar names (>80% match) will be assigned the same ID across segments.

## Legislation ID Format

Legislation IDs are generated using format: `L_BILLNUMBER_N`

Example: `L_HR1234_1`

- `L_` = Legislation prefix
- `BILLNUMBER` = Extracted bill number (e.g., HR1234, SB5678)
- `N` = Sequential number for duplicate bill names

**Detection Sources**: `audio`, `visual`, or `both`

## Video URL Configuration

To transcribe a different video, update the `TestVideo` class:

```python
class TestVideo(Video):
    YOUR_VIDEO = url_for_youtube_id("YOUR_YOUTUBE_ID")
```

Or use Cloud Storage:
```python
class TestVideo(Video):
    YOUR_VIDEO = "gs://your-bucket/path/to/video.mp4"
```

## Configuration Options

### Media Resolution
Currently set to `MEDIA_RESOLUTION_LOW` (66 tokens/frame) for faster processing with longer videos.

### Frame Rate
Currently set to `0.2 FPS` for optimal performance with 30-minute segments.

### Model
Currently using `GEMINI_2_5_FLASH` for best balance of speed and quality.

### Max Output Tokens
Set to `32,768` to handle long transcriptions within token limits.

## Troubleshooting

### `ValueError: Missing key inputs argument!`
Set your API credentials in environment variables (see Configuration section).

### `ImportError: No module named 'rapidfuzz'`
Install dependencies: `pip install yt-dlp rapidfuzz`

### `yt-dlp failed`
The script will fall back to filename parsing for video duration. Ensure your video enum name contains duration hint: `YOUR_VIDEO_PT1H30M`

### Token Limit Reached
The script handles segments up to 32K output tokens. For longer segments, reduce `--segment-minutes`.

### Speaker ID Mismatch
If speakers are incorrectly merged, adjust fuzzy matching threshold in `generate_speaker_id()` function (currently 80%).

## Performance Tips

1. **Use Overlap**: 1-2 minutes overlap ensures speaker continuity
2. **Segment Size**: 20-30 minutes per segment balances token usage and accuracy
3. **Order Context**: Providing the order paper helps with terminology and names
4. **Testing**: Test with `--max-segments 2` before processing full video

## Knowledge Graph Extraction

Extract knowledge graph entities and relationships from parliamentary transcripts using LLM-first approach with PostgreSQL backend.

### Features

- **LLM-First Extraction**: Uses Google Gemini to directly extract entities and relationships in a single pass
- **Semantic Edges**: Captures substantive relationships (AMENDS, GOVERNS, PROPOSES, etc.) and discourse relationships (RESPONDS_TO, AGREES_WITH, etc.)
- **Concept Nodes**: Extracts abstract concepts, policies, and systems (not just named entities)
- **Canonical IDs**: Stable hash-based node and edge IDs across runs
- **Vector Context**: Uses pgvector similarity to retrieve relevant known nodes for each window
- **Timestamp Accuracy**: Extracts timestamps from specific utterances referenced by each edge
- **Provenance**: Every edge includes evidence quotes, utterance IDs, and timestamps

### Architecture

**Window-Based Extraction**:
- Concept windows: 10 utterances with 6-utterance stride (40% overlap)
- Single LLM pass: Extracts both conceptual and discourse relationships together
- Context-aware: Top-K vector search provides relevant known nodes to LLM

**Database Schema**:
- `kg_nodes`: Canonical nodes with embeddings (speakers, legislation, concepts, orgs, places)
- `kg_aliases`: Normalized alias index for entity linking
- `kg_edges`: Canonical edges with provenance (evidence, utterance_ids, timestamps, confidence)

### Relationship Types

**Conceptual Relationships** (11 predicates):
- `AMENDS`: A legislation or policy amends another
- `GOVERNS`: Something governs or regulates a domain
- `MODERNIZES`: Updates or modernizes a system/process
- `AIMS_TO_REDUCE`: Goal to reduce something
- `REQUIRES_APPROVAL`: Needs approval from someone/something
- `IMPLEMENTED_BY`: Something is implemented by an entity
- `RESPONSIBLE_FOR`: Entity has responsibility
- `ASSOCIATED_WITH`: General association
- `CAUSES`: Causal relationship
- `ADDRESSES`: Addresses a topic/problem
- `PROPOSES`: Proposes legislation, policy, or idea

**Discourse Relationships** (4 predicates):
- `RESPONDS_TO`: Speaker responds to another speaker
- `AGREES_WITH`: Speaker expresses agreement
- `DISAGREES_WITH`: Speaker expresses disagreement
- `QUESTIONS`: Speaker asks a question

### Usage

```bash
# Extract KG from a single video
python scripts/kg_extract_from_video.py --youtube-video-id "Syxyah7QIaM"

# With custom window size and stride
python scripts/kg_extract_from_video.py --youtube-video-id "Syxyah7QIaM" --window-size 15 --stride 10

# Limit windows for testing
python scripts/kg_extract_from_video.py --youtube-video-id "Syxyah7QIaM" --max-windows 10

# With debug mode (save failed responses)
python scripts/kg_extract_from_video.py --youtube-video-id "Syxyah7QIaM" --debug
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|--------|----------|-------------|
| `--youtube-video-id` | str | Required | YouTube video ID to process |
| `--window-size` | int | 10 | Number of utterances per window |
| `--stride` | int | 6 | Utterances to advance between windows |
| `--max-windows` | int | None | Maximum windows to process (for testing) |
| `--run-id` | str | Auto-generated | KG run ID for traceability |
| `--model` | str | gemini-2.5-flash | Gemini model to use |
| `--top-k` | int | 25 | Top K candidate nodes to retrieve |
| `--no-filter-short` | flag | False | Don't filter short utterances |
| `--debug` | flag | False | Save failed responses to file |

### Output Statistics

After extraction, you'll see:
```
============================================================
Summary
============================================================
Windows processed: 162
  Successful: 158
  Failed: 4
New nodes: 1250
Edges: 2340
Links to known nodes: 890
Run ID: 075d3cd3-92f4-40a6-874a-f76a02ce1776
============================================================
```

### Requirements

```bash
pip install google-genai psycopg[binary] pgvector numpy
```

Set environment variable:
```bash
export GOOGLE_API_KEY="your-api-key"
```

### Database Setup

Ensure PostgreSQL tables exist:
- `kg_nodes`: id, label, type, aliases, embedding, created_at, updated_at
- `kg_aliases`: alias_norm, alias_raw, node_id, type, source, confidence
- `kg_edges`: id, source_id, predicate, target_id, youtube_video_id, earliest_timestamp_str, earliest_seconds, utterance_ids, evidence, speaker_ids, confidence, extractor_model, kg_run_id, created_at
      "speaker_id": "s_hon_santia_bradshaw_1",
      "timestamp": "00:36:00"
    }
  ],
  "statistics": {
    "total_nodes": 251,
    "total_edges": 262,
    "entity_types": {
      "PERSON": 18,
      "ORG": 53,
      "LAW": 3,
      "LEGISLATION": 3
    },
    "relation_types": {
      "DISCUSSES": 199,
      "PROPOSES": 16,
      "MENTIONS": 12
    }
  }
}
```

#### HTML Visualization

Open `knowledge_graph.html` in a browser to explore the interactive graph:
- **Physics-based layout**: Nodes naturally cluster by connections
- **Color-coded entities**: Different entity types have different colors
- **Interactive**: Click nodes to see connections, hover for details
- **Search**: Use the search box to find specific entities

### Example Output Statistics

```
Statistics:
  Total nodes: 251
  Total edges: 262

Entity types:
  ORG: 53
  DATE: 89
  CARDINAL: 37
  TIME: 21
  PERSON: 18
  ORDINAL: 3
  FAC: 13
  GPE: 12
  NORP: 5
  LAW: 3
  MONEY: 12
  QUANTITY: 2
  LOC: 3
  EVENT: 1
  PRODUCT: 2
  WORK_OF_ART: 1

Relation types:
  DISCUSSES: 199
  PROPOSES: 16
  MENTIONS: 12
  REFERENCES: 13
  WORKS_WITH: 10
  CRITICIZES: 7
  ADVOCATES_FOR: 3
  QUESTIONS: 2
```
