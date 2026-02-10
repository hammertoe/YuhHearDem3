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

Build a knowledge graph from transcript data using spaCy for entity extraction and Google Gemini for relationship extraction.

### Features

- **Entity Extraction**: Uses spaCy (en_core_web_md) to extract standard entities (PERSON, ORG, GPE, LAW, etc.)
- **Relationship Extraction**: Uses Google Gemini to extract relationships between entities
- **Date Normalization**: Automatically normalizes relative dates (e.g., "today", "last year") to absolute dates using video's upload date as anchor
- **Interactive Visualization**: Generates HTML visualization using NetworkX and PyVis
- **Structured Output**: Exports knowledge graph as JSON for further processing
- **Speaker & Legislation Nodes**: Automatically adds speakers and legislation as graph nodes

### Relationship Types

The script extracts the following predefined relationships:
- `MENTIONS`: Speaker A mentions Speaker B or an entity
- `REFERENCES`: Speaker references a law, bill, or legislation
- `AGREES_WITH`: Speaker expresses agreement with someone or something
- `DISAGREES_WITH`: Speaker expresses disagreement with someone or something
- `QUESTIONS`: Speaker asks a question to someone
- `RESPONDS_TO`: Speaker responds to someone
- `DISCUSSES`: Speaker discusses a topic
- `ADVOCATES_FOR`: Speaker supports or promotes something
- `CRITICIZES`: Speaker criticizes someone or something
- `PROPOSES`: Speaker proposes an idea or legislation
- `WORKS_WITH`: Collaborative relationship between entities

### Date Normalization

The script automatically normalizes relative date entities to absolute values using the video's upload date as an anchor point.

**Examples of normalized dates:**
- "today" → "2026-01-06" (using video upload date)
- "last year" → "2025"
- "last December" → "2025-12"
- "next week" → "2026-01-13" (approximately)

**Requirements:**
- Valid `video_metadata.upload_date` in transcript JSON (format: "YYYYMMDD")
- Dates must be within reasonable range (±100 years from current date)

**Output format:**
```json
{
  "id": "ent_abc123",
  "text": "last year",
  "type": "DATE",
  "resolved_date": "2025",
  "is_relative": true
}
```

**Limitations:**
- Failed resolutions are skipped silently
- Ambiguous dates use most recent interpretation
- Edge cases like "the 90s" or "Easter Monday" remain as textual labels

### Usage

```bash
# Basic usage
python build_knowledge_graph.py

# With custom files
python build_knowledge_graph.py --input-file transcription_output.json --output-json my_kg.json --output-html my_kg.html

# Adjust batch size (number of transcript entries per progress update)
python build_knowledge_graph.py --batch-size 20
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|--------|----------|-------------|
| `--input-file` | str | transcription_output.json | Input transcript JSON file |
| `--output-json` | str | knowledge_graph.json | Output JSON file for knowledge graph |
| `--output-html` | str | knowledge_graph.html | Output HTML file for visualization |
| `--gemini-api-key` | str | None | Gemini API key (uses GOOGLE_API_KEY env var if not set) |
| `--batch-size` | int | 10 | Progress update frequency |

### Requirements

```bash
pip install spacy spacy-llm networkx pyvis dateparser
python -m spacy download en_core_web_md
```

### Output Format

#### JSON Output
```json
{
  "nodes": [
    {
      "id": "ent_abc123",
      "text": "Hon Santia Bradshaw",
      "type": "PERSON",
      "speaker_context": ["s_hon_santia_bradshaw_1"],
      "timestamps": ["00:36:00"],
      "count": 15
    },
    {
      "id": "ent_def456",
      "text": "last year",
      "type": "DATE",
      "resolved_date": "2025",
      "is_relative": true,
      "speaker_context": ["s_hon_santia_bradshaw_1"],
      "timestamps": ["00:37:00"],
      "count": 5
    }
  ],
  "edges": [
    {
      "source": "ent_abc123",
      "target": "ent_def456",
      "relationship": "PROPOSES",
      "context": "...",
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
