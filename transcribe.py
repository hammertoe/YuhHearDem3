import argparse
import enum
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

import pydantic
import tenacity
import yt_dlp
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from google.genai.types import (
    FileData,
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    MediaResolution,
    Part,
    ThinkingConfig,
    VideoMetadata,
)
from rapidfuzz import fuzz

from lib.db.postgres_client import PostgresClient
from lib.gemini_finish_reason import (
    RetryableFinishReasonError,
    raise_if_retryable_finish_reason,
)
from lib.utils.config import config

load_dotenv()
client = genai.Client()


SamplingFrameRate = float
NOT_FOUND = "?"
YOUTUBE_URL_PREFIX = "https://www.youtube.com/watch?v="
CLOUD_STORAGE_URI_PREFIX = "gs://"


class Model(enum.Enum):
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    DEFAULT = GEMINI_2_5_FLASH


DEFAULT_CONFIG = GenerateContentConfig(
    temperature=0.0,
    top_p=0.0,
    seed=42,
)


def url_for_youtube_id(youtube_id: str) -> str:
    return f"{YOUTUBE_URL_PREFIX}{youtube_id}"


class TestVideo(enum.Enum):
    GDM_PODCAST_TRAILER_PT59S = url_for_youtube_id("0pJn3g8dfwk")
    JANE_GOODALL_PT2M42S = "gs://cloud-samples-data/video/JaneGoodall.mp4"
    GDM_ALPHAFOLD_PT7M54S = url_for_youtube_id("gg7WjuFs8F4")
    MY_VIDEO_PT10M = url_for_youtube_id("Syxyah7QIaM")


@dataclass(frozen=True)
class DynamicVideo:
    """Runtime-specified video compatible with Video API."""

    name: str
    value: str


Video = TestVideo | DynamicVideo


@dataclass
class VideoSegment:
    start: timedelta
    end: timedelta


VIDEO_TRANSCRIPTION_PROMPT = f"""
**Additional Context**

{{order_context}}

**Video Metadata**
- Title: {{video_title}}
- Date: {{video_date}}

**Known Speakers (from previous segments)**
Use these stable speaker IDs and names if you recognize the same person (voice IDs can change by segment):
{{known_speakers_context}}

**Previous Context (last 5 sentences)**
{{recent_context}}

**Reference Captions (same time window)**
Use this as a rough reference only. If it conflicts with the audio/video, prefer the audio/video.
{{caption_context}}

**Task 1 - Transcripts**

- Watch the video and listen carefully to the audio.
- Identify the distinct voices using a `voice` ID (1, 2, 3, etc.).
- Transcribe the video's audio verbatim with voice diarization.
- Break each speaker's speech into complete sentences.
- Include the absolute `start` timecode from the beginning of the video ({{timecode_spec}}) for each sentence segment.

**Task 2 - Speakers**

- For each `voice` ID from Task 1, extract information about the corresponding speaker.
- Use visual and audio cues to identify name and position.
- If a piece of information cannot be found, use `{NOT_FOUND}` as the value.
- Match with known speakers if the name is similar.
- Normalize speaker names consistently across segments (lowercase, no apostrophes/punctuation).

**Task 3 - Legislation**

- Extract any bills, laws, or legislation mentioned in the audio.
- Look for on-screen text displaying bill numbers or titles.
- Examples: "HR 1234", "Senate Bill 5678", "House Resolution 999".
- Provide a brief description of what the legislation is about.
- Note if it was detected from audio, visual, or both.
"""


class Transcript(pydantic.BaseModel):
    start: str
    text: str
    voice: int
    speaker_id: str | None = None


class Legislation(pydantic.BaseModel):
    id: str
    name: str
    description: str = ""
    source: str = "unknown"


class SpeakerEnhanced(pydantic.BaseModel):
    voice: int
    name: str
    position: str
    role_in_video: str
    speaker_id: str


class VideoTranscriptionEnhanced(pydantic.BaseModel):
    task1_transcripts: list[Transcript] = pydantic.Field(default_factory=list)
    task2_speakers: list[SpeakerEnhanced] = pydantic.Field(default_factory=list)
    task3_legislation: list[Legislation] = pydantic.Field(default_factory=list)
    video_title: str = ""
    video_date: str = ""


@dataclass
class VideoMetadataInfo:
    duration: timedelta
    title: str
    upload_date: str


def get_generate_content_config(model: Model, video: Video) -> GenerateContentConfig:
    media_resolution = get_media_resolution_for_video(video)
    thinking_config = get_thinking_config(model)

    return GenerateContentConfig(
        temperature=DEFAULT_CONFIG.temperature,
        top_p=DEFAULT_CONFIG.top_p,
        seed=DEFAULT_CONFIG.seed,
        response_mime_type="application/json",
        response_schema=VideoTranscriptionEnhanced,
        media_resolution=media_resolution,
        thinking_config=thinking_config,
        max_output_tokens=262144,
    )


def get_video_metadata_ytdlp(video: Video) -> VideoMetadataInfo:
    """Fetch video metadata using yt-dlp."""
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(video.value, download=False)

    duration_seconds = info.get("duration", 0) or 0
    return VideoMetadataInfo(
        duration=timedelta(seconds=duration_seconds),
        title=info.get("title") or "Unknown",
        upload_date=info.get("upload_date") or "Unknown",
    )


def detect_video_duration(video: Video) -> timedelta:
    """Detect actual video duration with fallbacks."""
    try:
        metadata = get_video_metadata_ytdlp(video)
        return metadata.duration
    except Exception as e:
        print(f"⚠️ yt-dlp failed: {e}")
        print("Falling back to filename parsing...")
        result = get_video_duration(video)
        return result or timedelta(minutes=60)


def get_video_duration(video: Video) -> timedelta | None:
    # For testing purposes, video duration is statically specified in the enum name
    # Suffix (ISO 8601 based): _PT[<h>H][<m>M][<s>S]
    # For production,
    # - fetch durations dynamically or store them separately
    # - take into account video VideoMetadata.start_offset & VideoMetadata.end_offset
    regex = r"_PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$"
    if not (match := re.search(regex, video.name)):
        print(f"⚠️ No duration info in {video.name}. Will use defaults.")
        return None

    h_str, m_str, s_str = match.groups()
    return timedelta(hours=int(h_str or 0), minutes=int(m_str or 0), seconds=int(s_str or 0))


def generate_speaker_id(name: str, existing_ids: dict[str, dict]) -> str:
    """Generate speaker ID: s_<name> (LLM normalizes names)"""

    for existing_id, existing_data in existing_ids.items():
        name_similarity = fuzz.ratio(name.lower(), existing_data["name"].lower())

        if name_similarity > 80:
            return existing_id

    normalized = name.lower().replace(" ", "_").replace("'", "")
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    base_id = f"s_{normalized}"

    counter = 1
    while f"{base_id}_{counter}" in existing_ids:
        counter += 1

    return f"{base_id}_{counter}"


def generate_legislation_id(name: str, existing_ids: set[str]) -> str:
    """Generate bill ID: L_BILLNUMBER_N"""
    bill_pattern = r"(?:HR|SB|HB|S|H)\s*\d+"
    match = re.search(bill_pattern, name.upper())

    if match:
        bill_number = match.group(0).replace(" ", "")
    else:
        bill_number = name.upper()[:10].replace(" ", "_")

    base_id = f"L_{bill_number}"
    counter = 1

    while f"{base_id}_{counter}" in existing_ids:
        counter += 1

    return f"{base_id}_{counter}"


def format_known_speakers(known_speakers_by_id: dict[str, dict]) -> str:
    """Format known speakers for prompt context."""
    if not known_speakers_by_id:
        return "No previous speakers identified."

    lines = []
    for speaker_id, speaker_data in known_speakers_by_id.items():
        lines.append(f"- {speaker_id}: {speaker_data['name']}")

    return "\n".join(lines)


def resolve_segment_speaker_mapping(
    segment_speakers: list[SpeakerEnhanced],
    known_speakers_by_id: dict[str, dict],
) -> tuple[dict[int, str], dict[str, int]]:
    """Resolve segment-local voice IDs to stable cross-segment speaker IDs."""
    voice_to_speaker_id: dict[int, str] = {}
    speaker_voice_samples: dict[str, int] = {}

    for speaker in segment_speakers:
        best_match_id = None
        best_score = 0
        speaker_name_lc = speaker.name.lower()

        for known_speaker_id, known_data in known_speakers_by_id.items():
            known_name = str(known_data.get("name", "")).lower()
            if not known_name:
                continue
            score = fuzz.ratio(speaker_name_lc, known_name)
            if score > best_score:
                best_score = score
                best_match_id = known_speaker_id

        if best_match_id and best_score > 80:
            speaker_id = best_match_id
        else:
            speaker_id = generate_speaker_id(
                speaker.name,
                {sid: data for sid, data in known_speakers_by_id.items() if data.get("name")},
            )

        speaker.speaker_id = speaker_id
        known_speakers_by_id[speaker_id] = {
            "name": speaker.name,
            "position": speaker.position,
            "role_in_video": speaker.role_in_video,
            "speaker_id": speaker_id,
        }
        voice_to_speaker_id[speaker.voice] = speaker_id
        speaker_voice_samples[speaker_id] = speaker.voice

    return voice_to_speaker_id, speaker_voice_samples


def get_previous_context(transcripts: list[Transcript], count: int = 5) -> str:
    """Get last N sentences as context string."""
    if not transcripts:
        return "No previous context."

    recent = transcripts[-count:]
    lines = []
    for t in recent:
        lines.append(f"[{t.start}] Voice {t.voice}: {t.text}")

    return "\n".join(lines)


def parse_timecode_to_timedelta(value: str) -> timedelta:
    parts = value.split(":")
    if len(parts) == 4:
        hours, mins, secs, millis = parts
    elif len(parts) == 3:
        hours, mins, secs = parts
        millis = "0"
        if float(secs) >= 60 and float(secs) < 1000:
            hours = "0"
            mins, secs, millis = parts
    elif len(parts) == 2:
        hours = "0"
        mins, secs = parts
        millis = "0"
    else:
        raise ValueError(f"Unexpected timecode format: {value}")

    mins_value = float(mins)
    secs_value = float(secs)
    millis_value = float(millis)

    if secs_value >= 60:
        raise ValueError(f"Invalid timecode values: {value}")

    if len(parts) >= 3 and mins_value >= 60:
        if float(hours) == 0:
            hours = str(int(mins_value // 60))
            mins_value = mins_value % 60
        else:
            raise ValueError(f"Invalid timecode values: {value}")

    if millis_value >= 1000:
        raise ValueError(f"Invalid millisecond value: {value}")

    return timedelta(
        hours=float(hours),
        minutes=mins_value,
        seconds=secs_value,
        milliseconds=millis_value,
    )


def format_timedelta_as_timecode(value: timedelta) -> str:
    total_seconds = int(value.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def normalize_caption_text(text: str) -> str:
    normalized = re.sub(r"<[^>]+>", " ", text)
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", normalized.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def clean_vtt_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def caption_similarity_score(left: str, right: str) -> float:
    left_norm = normalize_caption_text(left)
    right_norm = normalize_caption_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return float(fuzz.token_set_ratio(left_norm, right_norm))


@dataclass(frozen=True)
class CaptionGuardrailResult:
    status: str
    similarity: float
    reason: str

    def as_dict(self) -> dict[str, str | float]:
        return {"status": self.status, "similarity": self.similarity, "reason": self.reason}


def build_transcript_guardrail_text(
    transcripts: list[Transcript],
    *,
    max_seconds: int | None,
) -> str:
    if max_seconds is None:
        return " ".join(t.text for t in transcripts)

    filtered = [
        t.text
        for t in transcripts
        if parse_timecode_to_timedelta(t.start).total_seconds() <= max_seconds
    ]
    return " ".join(filtered)


def validate_transcript_against_captions(
    transcripts: list[Transcript],
    captions_text: str,
    *,
    min_similarity: float,
    max_seconds: int | None,
) -> CaptionGuardrailResult:
    if not captions_text.strip():
        return CaptionGuardrailResult(
            status="no_captions",
            similarity=0.0,
            reason="No captions available for comparison.",
        )

    transcript_text = build_transcript_guardrail_text(transcripts, max_seconds=max_seconds)
    if not transcript_text.strip():
        return CaptionGuardrailResult(
            status="empty_transcript",
            similarity=0.0,
            reason="Transcript is empty; cannot validate against captions.",
        )

    similarity = caption_similarity_score(transcript_text, captions_text)
    if similarity >= min_similarity:
        return CaptionGuardrailResult(
            status="ok",
            similarity=similarity,
            reason="Transcript matches captions within threshold.",
        )

    return CaptionGuardrailResult(
        status="mismatch",
        similarity=similarity,
        reason="Transcript diverges from captions beyond threshold.",
    )


def normalize_segment_transcript_timecodes(
    transcripts: list[Transcript],
    segment_start: timedelta,
    tolerance: timedelta = timedelta(seconds=2),
) -> list[Transcript]:
    if not transcripts:
        return transcripts

    min_time = min(parse_timecode_to_timedelta(t.start) for t in transcripts)
    if min_time < segment_start - tolerance:
        for transcript in transcripts:
            offset_time = parse_timecode_to_timedelta(transcript.start) + segment_start
            transcript.start = format_timedelta_as_timecode(offset_time)

    return transcripts


def deduplicate_transcripts(
    new_transcripts: list[Transcript],
    overlap_start: timedelta,
    overlap_end: timedelta,
    tolerance: timedelta = timedelta(seconds=2),
) -> list[Transcript]:
    """Remove transcripts in overlap period, keeping earliest."""
    filtered = []

    for transcript in new_transcripts:
        transcript_time = parse_timecode_to_timedelta(transcript.start)

        if transcript_time < overlap_start - tolerance:
            filtered.append(transcript)
        elif transcript_time > overlap_end + tolerance:
            filtered.append(transcript)

    return filtered


def get_media_resolution_for_video(video: Video) -> MediaResolution | None:
    return MediaResolution.MEDIA_RESOLUTION_LOW


def get_sampling_frame_rate_for_video(video: Video) -> SamplingFrameRate | None:
    return 0.2


def get_timecode_spec_for_model_and_video(model: Model, video: Video) -> str:
    timecode_spec = "H:MM:SS"

    match model:
        case Model.GEMINI_2_0_FLASH:
            pass
        case Model.GEMINI_2_5_FLASH | Model.GEMINI_2_5_PRO:
            pass
        case _:
            raise NotImplementedError(f"Undefined timecode spec for {model.name}.")

    return timecode_spec


def get_thinking_config(model: Model) -> ThinkingConfig | None:
    # Examples of thinking configurations (Gemini 2.5 models)
    match model:
        case Model.GEMINI_2_5_FLASH:  # Thinking disabled
            return ThinkingConfig(thinking_budget=0, include_thoughts=False)
        case Model.GEMINI_2_5_PRO:  # Minimum thinking budget and no summarized thoughts
            return ThinkingConfig(thinking_budget=128, include_thoughts=False)
        case _:
            return None  # Default


def get_video_part(
    video: Video,
    video_segment: VideoSegment | None = None,
    fps: float | None = None,
) -> Part | None:
    video_uri: str = video.value

    if not client.vertexai:
        video_uri = convert_to_https_url_if_cloud_storage_uri(video_uri)
        if not video_uri.startswith(YOUTUBE_URL_PREFIX):
            print("Google AI Studio API: Only YouTube URLs are currently supported")
            return None

    file_data = FileData(file_uri=video_uri, mime_type="video/*")
    video_metadata = get_video_part_metadata(video_segment, fps)

    return Part(file_data=file_data, video_metadata=video_metadata)


def get_video_part_metadata(
    video_segment: VideoSegment | None = None,
    fps: float | None = None,
) -> VideoMetadata:
    def offset_as_str(offset: timedelta) -> str:
        return f"{offset.total_seconds()}s"

    if video_segment:
        start_offset = offset_as_str(video_segment.start)
        end_offset = offset_as_str(video_segment.end)
    else:
        start_offset = None
        end_offset = None

    return VideoMetadata(start_offset=start_offset, end_offset=end_offset, fps=fps)


def parse_vtt_timecode(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours = "0"
        minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt_cues(vtt_text: str) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    current_start: float | None = None
    current_end: float | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_lines
        if current_start is None or current_end is None:
            return
        text = clean_vtt_text(" ".join(current_lines))
        if text:
            cues.append((current_start, current_end, text))
        current_start = None
        current_end = None
        current_lines = []

    for line in vtt_text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("WEBVTT") or stripped.startswith("Kind:"):
            continue
        if stripped.startswith("Language:"):
            continue
        if "-->" in stripped:
            flush()
            start, end = stripped.split("-->", maxsplit=1)
            current_start = parse_vtt_timecode(start.strip())
            current_end = parse_vtt_timecode(end.strip().split()[0])
            continue
        current_lines.append(stripped)

    flush()
    return cues


def build_caption_context(
    cues: list[tuple[float, float, str]],
    segment_start: timedelta,
    segment_end: timedelta,
    *,
    buffer_seconds: int,
    max_chars: int,
) -> str:
    start_sec = max(0.0, segment_start.total_seconds() - buffer_seconds)
    end_sec = segment_end.total_seconds() + buffer_seconds
    lines = [text for start, end, text in cues if start <= end_sec and end >= start_sec]
    if not lines:
        return ""
    context = " ".join(lines)
    if max_chars > 0:
        return context[:max_chars]
    return context


def extract_text_from_vtt(vtt_text: str, *, max_seconds: int | None) -> str:
    lines: list[str] = []
    current_start: float | None = None

    for line in vtt_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("WEBVTT") or stripped.startswith("Kind:"):
            continue
        if stripped.startswith("Language:"):
            continue
        if "-->" in stripped:
            start, _ = stripped.split("-->", maxsplit=1)
            current_start = parse_vtt_timecode(start.strip())
            continue
        if current_start is not None and max_seconds is not None:
            if current_start > max_seconds:
                continue
        lines.append(clean_vtt_text(stripped))

    return " ".join(lines)


def fetch_youtube_captions_vtt(video_url: str) -> str | None:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = os.path.join(temp_dir, "%(id)s")
            ydl_opts: dict[str, Any] = {
                "skip_download": True,
                "quiet": True,
                "no_warnings": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "vtt",
                "outtmpl": output_template,
            }
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(video_url, download=True)
                video_id = info.get("id") if isinstance(info, dict) else None
                if not video_id:
                    return None
                vtt_path = os.path.join(temp_dir, f"{video_id}.en.vtt")
                if not os.path.exists(vtt_path):
                    return None
                with open(vtt_path, encoding="utf-8") as handle:
                    return handle.read()
    except Exception as e:
        print(f"⚠️ Caption fetch failed: {e}")
        return None


def fetch_youtube_captions_text(
    video_url: str,
    *,
    max_seconds: int | None,
) -> str | None:
    vtt_text = fetch_youtube_captions_vtt(video_url)
    if not vtt_text:
        return None
    return extract_text_from_vtt(vtt_text, max_seconds=max_seconds)


def convert_to_https_url_if_cloud_storage_uri(uri: str) -> str:
    if uri.startswith(CLOUD_STORAGE_URI_PREFIX):
        return f"https://storage.googleapis.com/{uri.removeprefix(CLOUD_STORAGE_URI_PREFIX)}"
    else:
        return uri


def get_retrier() -> tenacity.Retrying:
    return tenacity.Retrying(
        stop=tenacity.stop_after_attempt(config.retry.max_attempts),
        wait=tenacity.wait_incrementing(
            start=config.retry.start_delay, increment=config.retry.delay_increment
        ),
        retry=should_retry_request,
        reraise=True,
    )


def should_retry_request(retry_state: tenacity.RetryCallState) -> bool:
    if not retry_state.outcome:
        return False
    err = retry_state.outcome.exception()
    if isinstance(err, RetryableFinishReasonError):
        print(f"❌ {err}")
        print("🔄 Retry: True")
        return True

    if not isinstance(err, ClientError):
        return False
    print(f"❌ ClientError {err.code}: {err.message}")

    retry = False
    match err.code:
        case 400 if err.message is not None and " try again " in err.message:
            retry = True
        case 429:
            retry = True
    print(f"🔄 Retry: {retry}")

    return retry


def display_response_info(response: GenerateContentResponse) -> None:
    if usage_metadata := response.usage_metadata:
        if usage_metadata.prompt_token_count:
            print(f"Input tokens   : {usage_metadata.prompt_token_count:9,d}")
        if usage_metadata.candidates_token_count:
            print(f"Output tokens  : {usage_metadata.candidates_token_count:9,d}")
        if usage_metadata.thoughts_token_count:
            print(f"Thoughts tokens: {usage_metadata.thoughts_token_count:9,d}")
    if not response.candidates:
        print("❌ No `response.candidates`")
        return
    if (finish_reason := response.candidates[0].finish_reason) != FinishReason.STOP:
        print(f"❌ {finish_reason = }")
    if not response.text:
        print("❌ No `response.text`")
        return


def print_combined_results(results: dict):
    """Print final combined transcription with IDs."""
    print("\n" + "=" * 80)
    print("TRANSCRIPTION")
    print("=" * 80)

    for t in results["transcripts"]:
        speaker_id = t.speaker_id or f"Voice_{t.voice}"
        print(f"[{t.start}] {speaker_id}: {t.text}")

    print("\n" + "=" * 80)
    print(f"SPEAKER REFERENCES ({len(results['speakers'])} speakers)")
    print("=" * 80)
    for speaker_id, info in results["speakers"].items():
        sample_voice_id = results.get("speaker_voice_samples", {}).get(speaker_id)
        print(f"\n{speaker_id}:")
        if sample_voice_id is not None:
            print(f"  Example Voice ID: {sample_voice_id}")
        print(f"  Name: {info['name']}")
        print(f"  Position: {info.get('position', NOT_FOUND)}")
        print(f"  Role: {info.get('role_in_video', NOT_FOUND)}")

    if results["legislation"]:
        print("\n" + "=" * 80)
        print(f"LEGISLATION REFERENCES ({len(results['legislation'])} items)")
        print("=" * 80)
        for leg in results["legislation"]:
            print(f"\n{leg.id}:")
            print(f"  Name: {leg.name}")
            if leg.description:
                print(f"  Description: {leg.description}")
            print(f"  Source: {leg.source}")

    print("\n" + "=" * 80)
    print("VIDEO METADATA")
    print("=" * 80)
    vm = results["video_metadata"]
    print(f"Title: {vm.title}")
    print(f"Upload Date: {vm.upload_date}")
    print(f"Duration: {vm.duration}")


def save_results_to_json(results: dict, output_file: str):
    """Save results to JSON file."""
    output_data = {
        "video_metadata": {
            "title": results["video_metadata"].title,
            "upload_date": results["video_metadata"].upload_date,
            "duration": str(results["video_metadata"].duration),
        },
        "transcripts": [
            {
                "start": t.start,
                "text": t.text,
                "voice_id": t.voice,
                "speaker_id": t.speaker_id,
            }
            for t in results["transcripts"]
        ],
        "speakers": [
            {
                "speaker_id": speaker_id,
                "voice_id": results.get("speaker_voice_samples", {}).get(speaker_id),
                "name": data["name"],
                "position": data.get("position", NOT_FOUND),
                "role_in_video": data.get("role_in_video", NOT_FOUND),
            }
            for speaker_id, data in results["speakers"].items()
        ],
        "legislation": [
            {
                "id": leg.id,
                "name": leg.name,
                "description": leg.description,
                "source": leg.source,
            }
            for leg in results["legislation"]
        ],
        "generated_at": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")


def get_video_transcription_segment(
    video: Video,
    segment_start: timedelta,
    segment_end: timedelta,
    order_text: str,
    known_speakers_context: str,
    recent_context: str,
    caption_context: str,
    video_metadata: VideoMetadataInfo,
) -> VideoTranscriptionEnhanced:
    """Process a single video segment."""
    model = Model.DEFAULT
    model_id = model.value

    fps = get_sampling_frame_rate_for_video(video)
    segment = VideoSegment(start=segment_start, end=segment_end)
    video_part = get_video_part(video, segment, fps)

    if not video_part:
        return VideoTranscriptionEnhanced()

    timecode_spec = get_timecode_spec_for_model_and_video(model, video)
    order_context = order_text if order_text else "No additional context provided."

    prompt = VIDEO_TRANSCRIPTION_PROMPT.format(
        timecode_spec=timecode_spec,
        order_context=order_context,
        video_title=video_metadata.title,
        video_date=video_metadata.upload_date,
        known_speakers_context=known_speakers_context,
        recent_context=recent_context,
        caption_context=caption_context or "No reference captions available.",
    )

    contents = [video_part, prompt.strip()]
    if order_text:
        contents.append(Part(text=order_text))

    config = get_generate_content_config(model, video)
    config = config.model_copy(update={"response_schema": VideoTranscriptionEnhanced})

    response = None
    for attempt in get_retrier():
        with attempt:
            response = client.models.generate_content(
                model=model_id, contents=contents, config=config
            )
            display_response_info(response)
            raise_if_retryable_finish_reason(response)

    if isinstance(response, GenerateContentResponse) and isinstance(
        response.parsed, VideoTranscriptionEnhanced
    ):
        return response.parsed

    print("❌ Could not parse the JSON response")
    return VideoTranscriptionEnhanced()


def process_video_iteratively(
    video: Video,
    segment_duration: int = 30,
    overlap_duration: int = 1,
    start_minutes: int = 0,
    max_segments: int | None = None,
    order_file: str | None = None,
    order_paper_id: str | None = None,
    validate_with_captions: bool = False,
    caption_guardrail_min_similarity: float = 45.0,
    caption_guardrail_mode: str = "warn",
    caption_guardrail_max_seconds: int | None = 1200,
    include_caption_context: bool = False,
    caption_context_max_chars: int = 1200,
    caption_context_buffer_seconds: int = 30,
) -> dict:
    """
    Process video in overlapping segments.
    Returns: {
        'transcripts': list[Transcript],
        'speakers': dict,
        'legislation': list[Legislation],
        'video_metadata': VideoMetadataInfo
    }
    """
    video_metadata = get_video_metadata_ytdlp(video)
    print(f"\n{'=' * 80}")
    print(f"Video: {video_metadata.title}")
    print(f"Date: {video_metadata.upload_date}")
    print(f"Duration: {video_metadata.duration}")
    print(f"{'=' * 80}\n")

    total_duration = video_metadata.duration
    segment_start = timedelta(minutes=start_minutes)
    all_transcripts = []
    known_speakers_by_id = {}
    speaker_voice_samples = {}
    all_legislation = []
    segment_num = 0

    order_text = ""
    if order_paper_id:
        order_text = get_order_paper_from_db(order_paper_id)
        if not order_text:
            print(f"⚠️ Order paper not found in DB: {order_paper_id}")
    elif order_file:
        with open(order_file) as f:
            order_text = f.read().strip()

    vtt_text: str | None = None
    caption_cues: list[tuple[float, float, str]] = []
    if validate_with_captions or include_caption_context:
        vtt_text = fetch_youtube_captions_vtt(video.value)
        if vtt_text:
            caption_cues = parse_vtt_cues(vtt_text)

    while segment_start < total_duration:
        if max_segments and segment_num >= max_segments:
            print(f"🛑 Reached max segments limit: {max_segments}")
            break

        segment_end = min(segment_start + timedelta(minutes=segment_duration), total_duration)

        print(f"\n{'─' * 80}")
        print(f"Segment {segment_num + 1}")
        print(f"Time: {segment_start} - {segment_end}")
        print(f"{'─' * 80}\n")

        known_speakers_context = format_known_speakers(known_speakers_by_id)
        recent_context = get_previous_context(all_transcripts, 5)
        caption_context = ""
        if include_caption_context and caption_cues:
            caption_context = build_caption_context(
                caption_cues,
                segment_start,
                segment_end,
                buffer_seconds=caption_context_buffer_seconds,
                max_chars=caption_context_max_chars,
            )

        result = None
        for attempt_num in range(3):
            try:
                result = get_video_transcription_segment(
                    video,
                    segment_start,
                    segment_end,
                    order_text,
                    known_speakers_context,
                    recent_context,
                    caption_context,
                    video_metadata,
                )
                break
            except Exception as e:
                print(f"❌ Attempt {attempt_num + 1} failed: {e}")
                if attempt_num == 2:
                    print(f"⚠️ Skipping segment {segment_num + 1}")
                    result = None
                    break

        if not result:
            segment_start = segment_end - timedelta(minutes=overlap_duration)
            segment_num += 1
            continue

        result.task1_transcripts = normalize_segment_transcript_timecodes(
            result.task1_transcripts, segment_start
        )

        if segment_num > 0:
            overlap_start = segment_start - timedelta(minutes=overlap_duration)
            overlap_end = segment_start
            result.task1_transcripts = deduplicate_transcripts(
                result.task1_transcripts, overlap_start, overlap_end
            )

        voice_to_speaker_id, segment_voice_samples = resolve_segment_speaker_mapping(
            result.task2_speakers,
            known_speakers_by_id,
        )
        speaker_voice_samples.update(segment_voice_samples)

        for transcript in result.task1_transcripts:
            transcript.speaker_id = voice_to_speaker_id.get(transcript.voice)

        existing_leg_ids = {leg_item.id for leg_item in all_legislation}
        for leg in result.task3_legislation:
            if not leg.id:
                leg_id = generate_legislation_id(leg.name, existing_leg_ids)
                leg.id = leg_id
                existing_leg_ids.add(leg_id)

            if leg.id not in {leg_item.id for leg_item in all_legislation}:
                all_legislation.append(leg)

        all_transcripts.extend(result.task1_transcripts)

        next_segment_start = segment_end - timedelta(minutes=overlap_duration)
        if next_segment_start <= segment_start:
            print("⚠️ Overlap caused backwards movement, stopping")
            print("🛑 End of video reached")
            break

        segment_start = next_segment_start
        segment_num += 1

    caption_guardrail: CaptionGuardrailResult | None = None
    if validate_with_captions:
        captions_text = None
        if vtt_text:
            captions_text = extract_text_from_vtt(
                vtt_text, max_seconds=caption_guardrail_max_seconds
            )
        caption_guardrail = validate_transcript_against_captions(
            all_transcripts,
            captions_text or "",
            min_similarity=caption_guardrail_min_similarity,
            max_seconds=caption_guardrail_max_seconds,
        )
        print(
            "✅ Caption guardrail status: "
            f"{caption_guardrail.status} (similarity={caption_guardrail.similarity:.2f})"
        )
        if caption_guardrail.status == "mismatch" and caption_guardrail_mode == "fail":
            raise ValueError("Caption guardrail mismatch; aborting transcript.")

    result = {
        "transcripts": all_transcripts,
        "speakers": known_speakers_by_id,
        "speaker_voice_samples": speaker_voice_samples,
        "legislation": all_legislation,
        "video_metadata": video_metadata,
    }

    if caption_guardrail:
        result["caption_guardrail"] = caption_guardrail.as_dict()

    return result


def get_order_paper_from_db(order_paper_id: str) -> str:
    """Fetch order paper from database and format as context text.

    Args:
        order_paper_id: Order paper ID

    Returns:
        Formatted order paper text for context
    """
    with PostgresClient() as postgres:
        result = postgres.execute_query(
            """
            SELECT session, sitting_date, sitting_number, parsed_json
            FROM order_papers
            WHERE id = %s
            """,
            (order_paper_id,),
        )

    if not result:
        return ""

    session, sitting_date, sitting_number, parsed_json = result[0]

    parsed: dict[str, object]
    if parsed_json is None:
        parsed = {}
    elif isinstance(parsed_json, str):
        try:
            parsed = json.loads(parsed_json)
        except Exception:
            parsed = {}
    elif isinstance(parsed_json, dict):
        parsed = parsed_json
    else:
        parsed = {}

    speakers = parsed.get("speakers")
    agenda_items = parsed.get("agenda_items")
    if not isinstance(speakers, list):
        speakers = []
    if not isinstance(agenda_items, list):
        agenda_items = []

    output: list[str] = []

    title = (session or "").strip() or f"ORDER PAPER {order_paper_id}"
    output.append("=" * 80)
    output.append(title.upper())
    output.append("=" * 80)
    if sitting_number:
        output.append(f"Sitting: {sitting_number}")
    if sitting_date:
        output.append(f"Date: {sitting_date}")
    output.append("")

    if speakers:
        output.append("SPEAKERS")
        output.append("-" * 40)
        for speaker in speakers:
            if not isinstance(speaker, dict):
                continue
            name = str(speaker.get("name", "")).strip()
            if not name:
                continue
            s_title = str(speaker.get("title", "")).strip()
            role = str(speaker.get("role", "")).strip()

            line = f"- {name}"
            if s_title:
                line += f" ({s_title})"
            if role:
                line += f" - {role}"
            output.append(line)
        output.append("")

    if agenda_items:
        output.append("AGENDA ITEMS")
        output.append("-" * 40)
        for idx, item in enumerate(agenda_items, 1):
            if not isinstance(item, dict):
                continue
            topic = str(item.get("topic_title", "")).strip()
            if not topic:
                continue
            mover = str(item.get("primary_speaker", "")).strip()
            description = str(item.get("description", "")).strip()

            output.append(f"{idx}. {topic}")
            if mover:
                output.append(f"   Mover: {mover}")
            if description:
                output.append(f"   Description: {description}")
            output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe video with Gemini (iterative processing)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="YouTube video ID, full YouTube URL, or gs:// URI (default: built-in test video)",
    )
    parser.add_argument(
        "--start-minutes",
        type=int,
        default=0,
        help="Start time in minutes (default: 0)",
    )
    parser.add_argument(
        "--order-file",
        type=str,
        default=None,
        help="Path to order file for additional context",
    )
    parser.add_argument(
        "--order-paper-id",
        type=str,
        default=None,
        help="Order paper ID from database (loads order paper for context)",
    )
    parser.add_argument(
        "--segment-minutes",
        type=int,
        default=30,
        help="Duration of each segment in minutes (default: 30)",
    )
    parser.add_argument(
        "--overlap-minutes",
        type=int,
        default=1,
        help="Overlap between segments in minutes (default: 1)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Maximum number of segments to process (default: all)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="transcription_output.json",
        help="Output JSON file path (default: transcription_output.json)",
    )
    parser.add_argument(
        "--validate-with-captions",
        action="store_true",
        help="Validate transcript against YouTube captions",
    )
    parser.add_argument(
        "--caption-guardrail-min-similarity",
        type=float,
        default=45.0,
        help="Minimum similarity required to pass caption guardrail (default: 45.0)",
    )
    parser.add_argument(
        "--caption-guardrail-max-seconds",
        type=int,
        default=1200,
        help="Seconds of transcript to compare in guardrail (default: 1200)",
    )
    parser.add_argument(
        "--caption-guardrail-mode",
        choices=("warn", "fail"),
        default="warn",
        help="Behavior when caption guardrail fails (warn|fail)",
    )
    parser.add_argument(
        "--caption-context",
        action="store_true",
        help="Include YouTube caption snippet in prompt as reference",
    )
    parser.add_argument(
        "--caption-context-max-chars",
        type=int,
        default=1200,
        help="Max characters of caption context to include (default: 1200)",
    )
    parser.add_argument(
        "--caption-context-buffer-seconds",
        type=int,
        default=30,
        help="Seconds of buffer around segment for caption context (default: 30)",
    )
    args = parser.parse_args()

    video: Video | DynamicVideo
    if args.video:
        raw = args.video.strip()
        if raw.startswith("http") or raw.startswith(CLOUD_STORAGE_URI_PREFIX):
            video_url = raw
            video_name = "DYNAMIC_VIDEO"
        else:
            video_url = url_for_youtube_id(raw)
            video_name = f"YOUTUBE_{raw}"
        video = DynamicVideo(name=video_name, value=video_url)
    else:
        video = TestVideo.MY_VIDEO_PT10M

    results = process_video_iteratively(
        video=video,
        segment_duration=args.segment_minutes,
        overlap_duration=args.overlap_minutes,
        start_minutes=args.start_minutes,
        max_segments=args.max_segments,
        order_file=args.order_file,
        order_paper_id=args.order_paper_id,
        validate_with_captions=args.validate_with_captions,
        caption_guardrail_min_similarity=args.caption_guardrail_min_similarity,
        caption_guardrail_mode=args.caption_guardrail_mode,
        caption_guardrail_max_seconds=args.caption_guardrail_max_seconds,
        include_caption_context=args.caption_context,
        caption_context_max_chars=args.caption_context_max_chars,
        caption_context_buffer_seconds=args.caption_context_buffer_seconds,
    )

    print_combined_results(results)
    save_results_to_json(results, args.output_file)
