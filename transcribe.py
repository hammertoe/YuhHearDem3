import argparse
import enum
import json
import os
import re
from collections.abc import Callable
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# To run this script, set one of the following environment variables:
# Option A - Google AI Studio API key:
#   export GOOGLE_API_KEY="your-api-key"
#
# Option B - Vertex AI (Google Cloud):
#   export GOOGLE_GENAI_USE_VERTEXAI="True"
#   export GOOGLE_CLOUD_PROJECT="your-project-id"
#   export GOOGLE_CLOUD_LOCATION="your-location"
#
# Get a Gemini API key from: https://aistudio.google.com/app/apikey
from datetime import timedelta

import pydantic
import tenacity
from google import genai
from google.genai.errors import ClientError
from google.genai.types import (
    FileData,
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    Part,
    VideoMetadata,
)
from google.genai.types import MediaResolution, ThinkingConfig

from dataclasses import dataclass

import yt_dlp
from rapidfuzz import fuzz


def check_environment() -> bool:
    check_colab_user_authentication()
    return check_manual_setup() or check_vertex_ai() or check_colab() or check_local()


def check_manual_setup() -> bool:
    return check_define_env_vars(
        False,
        "",
        "",
        "",
    )


def check_vertex_ai() -> bool:
    return False


def check_colab() -> bool:
    return False


def check_local() -> bool:
    vertexai, project, location, api_key = get_vars(os.getenv)
    return check_define_env_vars(vertexai, project, location, api_key)


def check_colab_user_authentication() -> None:
    pass


def get_vars(getenv: Callable[[str, str], str]) -> tuple[bool, str, str, str]:
    vertexai_str = getenv("GOOGLE_GENAI_USE_VERTEXAI", "")
    if vertexai_str:
        vertexai = vertexai_str.lower() in ["true", "1"]
    else:
        vertexai = bool(getenv("GOOGLE_CLOUD_PROJECT", ""))

    project = getenv("GOOGLE_CLOUD_PROJECT", "") if vertexai else ""
    location = getenv("GOOGLE_CLOUD_LOCATION", "") if project else ""
    api_key = getenv("GOOGLE_API_KEY", "") if not project else ""

    return vertexai, project, location, api_key


def check_define_env_vars(
    vertexai: bool,
    project: str,
    location: str,
    api_key: str,
) -> bool:
    match (vertexai, bool(project), bool(location), bool(api_key)):
        case (True, True, _, _):
            location = location or "global"
            define_env_vars(vertexai, project, location, "")
        case (True, False, _, True):
            define_env_vars(vertexai, "", "", api_key)
        case (False, _, _, True):
            define_env_vars(vertexai, "", "", api_key)
        case _:
            return False

    return True


def define_env_vars(vertexai: bool, project: str, location: str, api_key: str) -> None:
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    if project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
    if location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
    if vertexai:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = str(vertexai)


check_environment()
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


class Video(enum.Enum):
    pass


class TestVideo(Video):
    GDM_PODCAST_TRAILER_PT59S = url_for_youtube_id("0pJn3g8dfwk")
    JANE_GOODALL_PT2M42S = "gs://cloud-samples-data/video/JaneGoodall.mp4"
    GDM_ALPHAFOLD_PT7M54S = url_for_youtube_id("gg7WjuFs8F4")
    MY_VIDEO_PT10M = url_for_youtube_id("Syxyah7QIaM")


@dataclass(frozen=True)
class DynamicVideo:
    """Runtime-specified video compatible with Video API."""

    name: str
    value: str


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
Use these speaker IDs and names if you recognize the same person:
{{known_speakers_context}}

**Previous Context (last 5 sentences)**
{{recent_context}}

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


class Speaker(pydantic.BaseModel):
    voice: int
    name: str
    position: str
    role_in_video: str


class VideoTranscription(pydantic.BaseModel):
    task1_transcripts: list[Transcript] = pydantic.Field(default_factory=list)
    task2_speakers: list[Speaker] = pydantic.Field(default_factory=list)


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
        response_schema=VideoTranscription,
        media_resolution=media_resolution,
        thinking_config=thinking_config,
        max_output_tokens=32768,
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
    return timedelta(
        hours=int(h_str or 0), minutes=int(m_str or 0), seconds=int(s_str or 0)
    )


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


def format_known_speakers(known_speakers: dict) -> str:
    """Format known speakers for prompt context."""
    if not known_speakers:
        return "No previous speakers identified."

    lines = []
    for voice_id, speaker_data in known_speakers.items():
        lines.append(
            f"- Voice {voice_id}: {speaker_data['name']} -> {speaker_data['speaker_id']}"
        )

    return "\n".join(lines)


def get_previous_context(transcripts: list[Transcript], count: int = 5) -> str:
    """Get last N sentences as context string."""
    if not transcripts:
        return "No previous context."

    recent = transcripts[-count:]
    lines = []
    for t in recent:
        lines.append(f"[{t.start}] Voice {t.voice}: {t.text}")

    return "\n".join(lines)


def deduplicate_transcripts(
    new_transcripts: list[Transcript],
    overlap_start: timedelta,
    overlap_end: timedelta,
    tolerance: timedelta = timedelta(seconds=2),
) -> list[Transcript]:
    """Remove transcripts in overlap period, keeping earliest."""
    filtered = []

    for transcript in new_transcripts:
        time_parts = transcript.start.split(":")
        if len(time_parts) == 3:
            hours, mins, secs = map(float, time_parts)
            transcript_time = timedelta(hours=hours, minutes=mins, seconds=secs)
        else:
            mins, secs = map(float, time_parts)
            transcript_time = timedelta(minutes=mins, seconds=secs)

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


def convert_to_https_url_if_cloud_storage_uri(uri: str) -> str:
    if uri.startswith(CLOUD_STORAGE_URI_PREFIX):
        return f"https://storage.googleapis.com/{uri.removeprefix(CLOUD_STORAGE_URI_PREFIX)}"
    else:
        return uri


def get_retrier() -> tenacity.Retrying:
    return tenacity.Retrying(
        stop=tenacity.stop_after_attempt(7),
        wait=tenacity.wait_incrementing(start=10, increment=1),
        retry=should_retry_request,
        reraise=True,
    )


def should_retry_request(retry_state: tenacity.RetryCallState) -> bool:
    if not retry_state.outcome:
        return False
    err = retry_state.outcome.exception()
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


def get_video_transcription_from_response(
    response: GenerateContentResponse,
) -> VideoTranscription:
    if isinstance(response.parsed, VideoTranscription):
        return response.parsed

    print("❌ Could not parse the JSON response")
    return VideoTranscription()  # Empty transcription


def get_video_transcription(
    video: Video,
    video_segment: VideoSegment | None = None,
    fps: float | None = None,
    prompt: str | None = None,
    model: Model | None = None,
    order_file: str | None = None,
) -> VideoTranscription:
    model = model or Model.DEFAULT
    model_id = model.value

    fps = fps or get_sampling_frame_rate_for_video(video)
    video_part = get_video_part(video, video_segment, fps)
    if not video_part:  # Unsupported source, return an empty transcription
        return VideoTranscription()

    if prompt is None:
        timecode_spec = get_timecode_spec_for_model_and_video(model, video)
        order_context = ""
        if order_file:
            order_context = "An order document is provided below for context."
        else:
            order_context = "No additional context provided."
        prompt = VIDEO_TRANSCRIPTION_PROMPT.format(
            timecode_spec=timecode_spec, order_context=order_context
        )

    contents = [video_part, prompt.strip()]
    if order_file:
        with open(order_file, "r") as f:
            order_text = f.read().strip()
        contents.append(Part(text=order_text))

    config = get_generate_content_config(model, video)

    print(f" {video.name} / {model_id} ".center(80, "-"))
    response = None
    for attempt in get_retrier():
        with attempt:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )
            display_response_info(response)

    assert isinstance(response, GenerateContentResponse)
    return get_video_transcription_from_response(response)


def print_combined_results(results: dict):
    """Print final combined transcription with IDs."""
    print("\n" + "=" * 80)
    print("TRANSCRIPTION")
    print("=" * 80)

    for t in results["transcripts"]:
        speaker = results["speakers"].get(t.voice, {})
        speaker_id = speaker.get("speaker_id", f"Voice_{t.voice}")
        print(f"[{t.start}] {speaker_id}: {t.text}")

    print("\n" + "=" * 80)
    print(f"SPEAKER REFERENCES ({len(results['speakers'])} speakers)")
    print("=" * 80)
    for voice_id, info in results["speakers"].items():
        print(f"\n{info['speaker_id']}:")
        print(f"  Voice ID: {voice_id}")
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
                "speaker_id": results["speakers"].get(t.voice, {}).get("speaker_id"),
            }
            for t in results["transcripts"]
        ],
        "speakers": [
            {
                "speaker_id": data["speaker_id"],
                "voice_id": voice_id,
                "name": data["name"],
                "position": data.get("position", NOT_FOUND),
                "role_in_video": data.get("role_in_video", NOT_FOUND),
            }
            for voice_id, data in results["speakers"].items()
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
    known_speakers = {}
    all_legislation = []
    segment_num = 0

    order_text = ""
    if order_file:
        with open(order_file, "r") as f:
            order_text = f.read().strip()

    while segment_start < total_duration:
        if max_segments and segment_num >= max_segments:
            print(f"🛑 Reached max segments limit: {max_segments}")
            break

        segment_end = min(
            segment_start + timedelta(minutes=segment_duration), total_duration
        )

        print(f"\n{'─' * 80}")
        print(f"Segment {segment_num + 1}")
        print(f"Time: {segment_start} - {segment_end}")
        print(f"{'─' * 80}\n")

        known_speakers_context = format_known_speakers(known_speakers)
        recent_context = get_previous_context(all_transcripts, 5)

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

        if segment_num > 0:
            overlap_start = segment_start - timedelta(minutes=overlap_duration)
            overlap_end = segment_start
            result.task1_transcripts = deduplicate_transcripts(
                result.task1_transcripts, overlap_start, overlap_end
            )

        known_by_name = {}
        for voice_id, speaker_data in known_speakers.items():
            speaker_name = speaker_data["name"].lower()
            if speaker_name not in known_by_name:
                known_by_name[speaker_name] = []
            known_by_name[speaker_name].append((voice_id, speaker_data))

        matched_speakers = {}
        for speaker in result.task2_speakers:
            matched = False
            for known_name, speaker_list in known_by_name.items():
                name_similarity = fuzz.ratio(speaker.name.lower(), known_name)
                if name_similarity > 80:
                    matched_speakers[speaker.voice] = speaker_list[0][1]
                    matched = True
                    break

            if not matched:
                speaker_id = generate_speaker_id(
                    speaker.name,
                    {v: d for v, d in known_speakers.items() if d.get("name")},
                )
                speaker.speaker_id = speaker_id
                known_speakers[speaker.voice] = {
                    "name": speaker.name,
                    "position": speaker.position,
                    "role_in_video": speaker.role_in_video,
                    "speaker_id": speaker_id,
                }

        for speaker in result.task2_speakers:
            if speaker.voice in matched_speakers:
                speaker.speaker_id = matched_speakers[speaker.voice]["speaker_id"]

        existing_leg_ids = {l.id for l in all_legislation}
        for leg in result.task3_legislation:
            if not leg.id:
                leg_id = generate_legislation_id(leg.name, existing_leg_ids)
                leg.id = leg_id
                existing_leg_ids.add(leg_id)

            if leg.id not in {l.id for l in all_legislation}:
                all_legislation.append(leg)

        all_transcripts.extend(result.task1_transcripts)

        next_segment_start = segment_end - timedelta(minutes=overlap_duration)
        if next_segment_start <= segment_start:
            print("⚠️ Overlap caused backwards movement, stopping")
            print("🛑 End of video reached")
            break

        consecutive_empty_segments = 0
        segment_start = next_segment_start
        segment_num += 1

    return {
        "transcripts": all_transcripts,
        "speakers": known_speakers,
        "legislation": all_legislation,
        "video_metadata": video_metadata,
    }


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
    )

    print_combined_results(results)
    save_results_to_json(results, args.output_file)
