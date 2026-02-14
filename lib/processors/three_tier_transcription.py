"""Three-tier transcription output processor."""

from __future__ import annotations

import json
import re
from typing import Any

from lib.processors.paragraph_splitter import (
    group_transcripts_into_paragraphs,
    split_paragraph_into_sentences,
)


class ThreeTierTranscriptionProcessor:
    """Process transcripts into three-tier format (paragraphs + sentences)."""

    def process_transcript_to_three_tier(
        self,
        youtube_video_id: str,
        video_title: str,
        video_date: str,
        transcripts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Process transcript data into three-tier format."""
        paragraphs = group_transcripts_into_paragraphs(youtube_video_id, transcripts)

        three_tier_data = {
            "video_metadata": {
                "youtube_id": youtube_video_id,
                "title": video_title,
                "upload_date": video_date,
                "processed_at": self.get_current_timestamp(),
            },
            "paragraphs": [],
            "sentences": [],
            "speakers": self._extract_speakers(transcripts),
            "legislation": self._extract_legislation(transcripts),
        }
        used_sentence_ids: set[str] = set()

        for paragraph in paragraphs:
            para_dict = paragraph.to_dict()

            para_dict["video_id"] = youtube_video_id
            para_dict["video_title"] = video_title
            para_dict["video_date"] = video_date

            three_tier_data["paragraphs"].append(para_dict)

            sentences = split_paragraph_into_sentences(
                paragraph,
                youtube_video_id,
                video_date,
                video_title,
                existing_sentence_ids=used_sentence_ids,
            )

            three_tier_data["sentences"].extend(sentences)

        return three_tier_data

    def _extract_speakers(self, transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract unique speakers from transcripts."""
        speakers_map: dict[str, dict[str, Any]] = {}

        def speaker_id_to_base_name(speaker_id: str) -> str:
            if speaker_id.startswith("s_"):
                base = speaker_id.removeprefix("s_")
                base = re.sub(r"_\d+$", "", base)
                return base
            return speaker_id

        for entry in transcripts:
            speaker_id = entry.get("speaker_id", "")
            if not speaker_id:
                continue

            base_name = speaker_id_to_base_name(speaker_id)
            if speaker_id not in speakers_map:
                speakers_map[speaker_id] = {
                    "speaker_id": speaker_id,
                    "normalized_name": base_name,
                    "full_name": base_name.replace("_", " "),
                    "title": "",
                    "position": "Unknown",
                    "role_in_video": "member",
                    "first_appearance": entry.get("start", ""),
                    "total_appearances": 0,
                }

            speakers_map[speaker_id]["total_appearances"] += 1

        speakers_list = list(speakers_map.values())
        return sorted(speakers_list, key=lambda x: x.get("first_appearance", ""))

    def _extract_legislation(self, transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract legislation mentions from transcripts."""
        leg_map: dict[str, dict[str, Any]] = {}

        for entry in transcripts:
            text = entry.get("text", "")

            # Prefer capturing the full titled bill phrase, e.g. "Road Traffic Bill".
            patterns = [
                (r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+Bill)\b", "BILL"),
                (r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+Act)\b", "LAW"),
            ]

            for pattern, leg_type in patterns:
                for match in re.finditer(pattern, text):
                    name = match.group(1).strip()
                    # Filter generic matches like "This bill".
                    if name.lower() in {"this bill", "the bill", "a bill"}:
                        continue
                    if name not in leg_map:
                        leg_map[name] = {
                            "id": "",
                            "name": name,
                            "type": leg_type,
                            "source": "audio",
                            "mentions": 0,
                        }
                    leg_map[name]["mentions"] += 1

        legislation_list = sorted(leg_map.values(), key=lambda x: x["mentions"], reverse=True)

        return legislation_list

    def get_current_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def save_three_tier_output(self, data: dict[str, Any], output_file: str) -> None:
        """Save three-tier transcription output."""
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved three-tier output to: {output_file}")
        print(f"   - Video: {data['video_metadata']['title']}")
        print(f"   - Paragraphs: {len(data['paragraphs'])}")
        print(f"   - Sentences: {len(data['sentences'])}")
        print(f"   - Speakers: {len(data['speakers'])}")
        print(f"   - Legislation: {len(data['legislation'])}")
