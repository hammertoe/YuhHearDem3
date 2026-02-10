#!/usr/bin/env python3
"""Build knowledge graph from parliamentary transcripts using spaCy and Gemini."""

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import networkx as nx
import spacy
from google import genai
from google.genai.types import GenerateContentConfig, FinishReason
from pyvis.network import Network
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
)

from lib.processors.paragraph_splitter import group_transcripts_into_paragraphs


@dataclass
class Entity:
    """Represents an entity extracted from transcripts."""

    id: str
    text: str
    type: str
    speaker_context: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    count: int = 1
    resolved_date: str | None = None
    is_relative: bool = False


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    source_id: str
    target_id: str
    relation: str
    context: str
    speaker_id: str
    timestamp: str


@dataclass
class KnowledgeGraph:
    """Represents the complete knowledge graph."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    statistics: dict[str, Any]


class GeminiModel:
    """Custom LLM model for spacy-llm using Google Gemini."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.model_name = model_name
        self.client = genai.Client()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def __call__(self, prompts: list[str]) -> list[str]:
        """Process a list of prompts and return responses."""
        responses = []
        for prompt in prompts:
            try:
                result = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=2048,
                    ),
                )
                if (
                    result.candidates
                    and result.candidates[0].finish_reason == FinishReason.STOP
                    and result.candidates[0].content
                    and result.candidates[0].content.parts
                    and result.candidates[0].content.parts[0].text
                ):
                    responses.append(result.candidates[0].content.parts[0].text)
                else:
                    responses.append("")
            except Exception as e:
                # Don't fail the whole run, but surface the error so we know we're losing edges.
                print(f"⚠️ Gemini relationship extraction error: {e}")
                responses.append("")
        return responses


class KnowledgeGraphBuilder:
    """Build knowledge graph from parliamentary transcripts."""

    RELATION_TYPES = [
        "MENTIONS",
        "REFERENCES",
        "AGREES_WITH",
        "DISAGREES_WITH",
        "QUESTIONS",
        "RESPONDS_TO",
        "DISCUSSES",
        "ADVOCATES_FOR",
        "CRITICIZES",
        "PROPOSES",
        "WORKS_WITH",
    ]

    # By default, exclude high-frequency numeric/time entities from relationship prompts.
    # They create many nodes but rarely produce useful semantic relationships.
    DEFAULT_RELATION_ENTITY_TYPES = {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "FAC",
        "NORP",
        "LAW",
        "EVENT",
        "PRODUCT",
        "WORK_OF_ART",
        "LEGISLATION",
        "SPEAKER",
    }

    def __init__(
        self,
        gemini_api_key: str,
        spacy_model: str = "en_core_web_md",
        *,
        nlp: Any | None = None,
        gemini_model: Any | None = None,
    ):
        self.nlp = nlp or spacy.load(spacy_model)
        self.gemini_model = gemini_model or GeminiModel(gemini_api_key)
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.speaker_map: dict[str, dict[str, Any]] = {}
        self._debug_prompts_printed = 0

    def load_transcript_data(self, filepath: str) -> dict[str, Any]:
        """Load transcript JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def normalize_entity_text(self, text: str) -> str:
        """Normalize entity text for deduplication."""
        return text.strip().lower()

    def _truncate_text_for_prompt(self, text: str, max_chars: int = 1800) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _select_entities_for_relationships(
        self,
        entities: list[tuple[str, str, str]],
        *,
        include_numeric_entities: bool,
        max_entities: int,
    ) -> list[dict[str, str]]:
        """Select and rank entities for relationship extraction prompts."""
        # Count mentions by entity id within this span.
        counts: dict[str, int] = {}
        first: dict[str, dict[str, str]] = {}
        for eid, etext, etype in entities:
            counts[eid] = counts.get(eid, 0) + 1
            if eid not in first:
                first[eid] = {"id": eid, "text": etext, "type": etype}

        selected = list(first.values())
        if not include_numeric_entities:
            selected = [
                e
                for e in selected
                if e.get("type") in self.DEFAULT_RELATION_ENTITY_TYPES
            ]

        type_priority = {
            "LAW": 0,
            "LEGISLATION": 0,
            "ORG": 1,
            "PERSON": 2,
            "GPE": 3,
            "LOC": 4,
            "FAC": 4,
            "NORP": 5,
            "SPEAKER": 6,
        }

        def sort_key(e: dict[str, str]) -> tuple[int, int, int]:
            eid = e["id"]
            pri = type_priority.get(e.get("type", ""), 10)
            return (pri, -counts.get(eid, 0), len(e.get("text", "")))

        selected.sort(key=sort_key)
        return selected[:max_entities]

    def generate_entity_id(self, text: str, entity_type: str) -> str:
        """Generate unique ID for an entity."""
        normalized = self.normalize_entity_text(text)
        unique_str = f"{entity_type}:{normalized}"
        hash_obj = hashlib.md5(unique_str.encode())
        return f"ent_{hash_obj.hexdigest()[:12]}"

    def _is_relative_date(self, text: str) -> bool:
        """Check if date text appears to be relative."""
        if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
            return False
        if re.match(r"^\d{4}$", text):
            return False

        relative_keywords = [
            "today",
            "yesterday",
            "tomorrow",
            "last",
            "next",
            "ago",
            "recent",
            "this",
            "current",
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in relative_keywords)

    def _format_resolved_date(self, date_obj: datetime) -> str:
        """Format resolved date based on precision."""
        if date_obj.day == 1:
            return date_obj.strftime("%Y-%m")
        else:
            return date_obj.strftime("%Y-%m-%d")

    def _is_reasonable_date(self, year: int) -> bool:
        """Validate that date year is reasonable."""
        current_year = datetime.now().year
        min_year = current_year - 100
        max_year = current_year + 10
        return min_year <= year <= max_year

    def _resolve_last_month_pattern(
        self, text: str, anchor_date: datetime
    ) -> datetime | None:
        """Resolve 'last [Month]' patterns manually."""
        month_names = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]

        text_lower = text.lower()

        for month_name in month_names:
            if month_name in text_lower:
                try:
                    month_num = datetime.strptime(month_name, "%B").month
                    return datetime(anchor_date.year - 1, month_num, 1)
                except ValueError:
                    pass

        return None

    def normalize_relative_dates(self, anchor_date: datetime):
        """Normalize relative date entities to absolute values."""
        from dateparser import parse

        for entity in self.entities.values():
            if entity.type != "DATE":
                continue

            if not self._is_relative_date(entity.text):
                continue

            try:
                resolved = parse(
                    entity.text,
                    settings={
                        "RELATIVE_BASE": anchor_date,
                        "DATE_ORDER": "YMD",
                        "STRICT_PARSING": False,
                    },
                )
                if not resolved:
                    resolved = self._resolve_last_month_pattern(
                        entity.text, anchor_date
                    )

                if resolved and self._is_reasonable_date(resolved.year):
                    entity.resolved_date = self._format_resolved_date(resolved)
                    entity.is_relative = True
                else:
                    entity.resolved_date = None
                    entity.is_relative = False
            except Exception:
                entity.resolved_date = None
                entity.is_relative = False

    def extract_entities_from_text(
        self, text: str, speaker_id: str, timestamp: str
    ) -> list[tuple[str, str, str]]:
        """Extract entities from a text span (sentence, paragraph, window)."""
        doc = self.nlp(text)
        extracted = []

        for ent in doc.ents:
            entity_id = self.generate_entity_id(ent.text, ent.label_)

            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    text=ent.text,
                    type=ent.label_,
                    speaker_context=[speaker_id],
                    timestamps=[timestamp],
                    count=1,
                )
            else:
                entity = self.entities[entity_id]
                if speaker_id not in entity.speaker_context:
                    entity.speaker_context.append(speaker_id)
                if timestamp not in entity.timestamps:
                    entity.timestamps.append(timestamp)
                entity.count += 1

            extracted.append((entity_id, ent.text, ent.label_))

        return extracted

    def extract_entities(
        self, transcript_entry: dict[str, Any]
    ) -> list[tuple[str, str, str]]:
        """Extract entities from a single transcript entry."""
        return self.extract_entities_from_text(
            transcript_entry.get("text", ""),
            transcript_entry.get("speaker_id", ""),
            transcript_entry.get("start", ""),
        )

    def extract_relationships(
        self,
        entities: list[tuple[str, str, str]],
        text: str,
        speaker_id: str,
        timestamp: str,
        *,
        include_numeric_entities: bool = False,
        max_entities: int = 20,
    ) -> list[Relationship]:
        """Extract relationships between entities using Gemini."""
        if len(entities) < 2:
            return []

        # De-dupe entities by id while preserving first occurrence.
        unique: dict[str, dict[str, str]] = {}
        for eid, etext, etype in entities:
            if eid not in unique:
                unique[eid] = {"id": eid, "text": etext, "type": etype}

        entity_list = list(unique.values())

        if not include_numeric_entities:
            entity_list = [
                e
                for e in entity_list
                if e.get("type") in self.DEFAULT_RELATION_ENTITY_TYPES
            ]

        # Keep prompts bounded.
        entity_list = entity_list[:max_entities]

        if len(entity_list) < 2:
            return []

        prompt = self._build_relationship_prompt(entity_list, text, speaker_id)

        debug = os.getenv("KG_DEBUG_PROMPTS") in {"1", "true", "TRUE", "yes", "YES"}
        if debug:
            try:
                max_prompts = int(os.getenv("KG_DEBUG_PROMPTS_MAX", "1"))
            except ValueError:
                max_prompts = 1

            if self._debug_prompts_printed < max_prompts:
                self._debug_prompts_printed += 1
                print("\n" + "=" * 80)
                print(f"KG Gemini Prompt ({self._debug_prompts_printed}/{max_prompts})")
                print("=" * 80)
                print(prompt)
                print("=" * 80 + "\n")

        try:
            response = self.gemini_model([prompt])[0]
            return self._parse_relationships(
                response, entity_list, text, speaker_id, timestamp
            )
        except Exception as e:
            print(f"Error extracting relationships: {e}")
            return []

    def _build_relationship_prompt(
        self, entities: list[dict[str, str]], text: str, speaker_id: str
    ) -> str:
        """Build prompt for relationship extraction."""
        # Use indices to make LLM output more reliable.
        entity_descriptions = "\n".join(
            f"- {i + 1}. {e['text']} (Type: {e['type']}, ID: {e['id']})"
            for i, e in enumerate(entities)
        )

        relation_descriptions = "\n".join(
            f"- {rel}: {self._get_relation_description(rel)}"
            for rel in self.RELATION_TYPES
        )

        prompt = f"""Analyze the following text and extract relationships between the entities.

Speaker: {speaker_id}

Text: {text}

Entities:
{entity_descriptions}

Relationship Types:
{relation_descriptions}

Instructions:
1. Identify relationships between entities based on the text
2. Only extract relationships that are explicitly stated or strongly implied
3. Output STRICT JSON only (no markdown, no code fences)
4. The JSON must be an array of objects with fields:
   - source_index: integer (from the numbered list)
   - relation: one of the allowed relationship types
   - target_index: integer (from the numbered list)
   - evidence: short text snippet from the passage
5. Prefer quality over quantity; output at most 12 relationships
6. If none, output []

Example output:
[
  {{"source_index": 1, "relation": "DISCUSSES", "target_index": 3, "evidence": "..."}}
]

JSON:
"""
        return prompt

    def _parse_relationships_json(
        self,
        response: str,
        entities: list[dict[str, str]],
        text: str,
        speaker_id: str,
        timestamp: str,
    ) -> list[Relationship]:
        """Parse JSON response containing source/target indices."""
        if not response:
            return []

        raw = response.strip()
        # Strip code fences if present.
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```\s*$", "", raw).strip()

        # Extract the first JSON array if the model included extra text.
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []

        raw_json = raw[start : end + 1]
        try:
            data = json.loads(raw_json)
        except Exception:
            return []

        if not isinstance(data, list):
            return []

        rel_types = set(self.RELATION_TYPES)
        relationships: list[Relationship] = []

        def idx_to_id(idx: int) -> str | None:
            if idx < 1 or idx > len(entities):
                return None
            return entities[idx - 1]["id"]

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                s_raw = item.get("source_index")
                t_raw = item.get("target_index")
                if s_raw is None or t_raw is None:
                    continue
                s_idx = int(s_raw)
                t_idx = int(t_raw)
            except Exception:
                continue

            relation = str(item.get("relation", "")).upper().strip()
            if relation not in rel_types:
                continue

            source_id = idx_to_id(s_idx)
            target_id = idx_to_id(t_idx)
            if not source_id or not target_id:
                continue

            relationships.append(
                Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    context=text[:200] + "..." if len(text) > 200 else text,
                    speaker_id=speaker_id,
                    timestamp=timestamp,
                )
            )

        return relationships

    def _get_relation_description(self, relation: str) -> str:
        """Get description for a relation type."""
        descriptions = {
            "MENTIONS": "Speaker A mentions Speaker B or an entity",
            "REFERENCES": "Speaker references a law, bill, or legislation",
            "AGREES_WITH": "Speaker expresses agreement with someone or something",
            "DISAGREES_WITH": "Speaker expresses disagreement with someone or something",
            "QUESTIONS": "Speaker asks a question to someone",
            "RESPONDS_TO": "Speaker responds to someone",
            "DISCUSSES": "Speaker discusses a topic",
            "ADVOCATES_FOR": "Speaker supports or promotes something",
            "CRITICIZES": "Speaker criticizes someone or something",
            "PROPOSES": "Speaker proposes an idea or legislation",
            "WORKS_WITH": "Collaborative relationship between entities",
        }
        return descriptions.get(relation, relation)

    def _parse_relationships(
        self,
        response: str,
        entities: list[dict[str, str]],
        text: str,
        speaker_id: str,
        timestamp: str,
    ) -> list[Relationship]:
        """Parse LLM response into Relationship objects."""
        relationships: list[Relationship] = []
        valid_ids = {e["id"] for e in entities}

        # Accept a few common formats and strip punctuation around ids.
        rel_types = set(self.RELATION_TYPES)
        line_re = re.compile(
            r"(ent_[0-9a-f]{12}|speaker_[^\s\)\]\}]+|leg_[^\s\)\]\}]+)\s+([A-Z_]+)\s+"
            r"(ent_[0-9a-f]{12}|speaker_[^\s\)\]\}]+|leg_[^\s\)\]\}]+)",
            flags=re.IGNORECASE,
        )

        for raw in (response or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            match = line_re.search(line)
            if not match:
                continue

            source_id = match.group(1).strip('()[]{}<>,."')
            relation = match.group(2).upper().strip('[](){}<>,."')
            target_id = match.group(3).strip('()[]{}<>,."')

            if relation not in rel_types:
                continue
            if source_id not in valid_ids or target_id not in valid_ids:
                continue

            relationships.append(
                Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    context=text[:200] + "..." if len(text) > 200 else text,
                    speaker_id=speaker_id,
                    timestamp=timestamp,
                )
            )

        return relationships

    def build_speaker_map(self, data: dict[str, Any]):
        """Build mapping from speaker IDs to speaker info."""
        speakers = data.get("speakers", [])
        for speaker in speakers:
            self.speaker_map[speaker["speaker_id"]] = speaker

    def add_speaker_nodes(self):
        """Add speaker nodes to the entity collection."""
        for speaker_id, speaker_info in self.speaker_map.items():
            entity_id = f"speaker_{speaker_id}"
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    text=speaker_info.get("name", speaker_id),
                    type="SPEAKER",
                    speaker_context=[speaker_id],
                    timestamps=[],
                    count=1,
                )

    def add_legislation_nodes(self, data: dict[str, Any]):
        """Add legislation nodes to the entity collection."""
        legislation = data.get("legislation", [])
        for leg in legislation:
            leg_id = f"leg_{leg['id']}"
            if leg_id not in self.entities:
                self.entities[leg_id] = Entity(
                    id=leg_id,
                    text=leg.get("name", leg["id"]),
                    type="LEGISLATION",
                    speaker_context=[],
                    timestamps=[],
                    count=1,
                )

    def build_knowledge_graph(
        self,
        data: dict[str, Any],
        batch_size: int = 10,
        *,
        granularity: str = "paragraph",
        youtube_video_id: str = "unknown_video",
        include_numeric_entities_in_relations: bool = False,
        max_entities_per_prompt: int = 20,
        add_speaker_mention_edges: bool = True,
        add_cooccurrence_edges: bool = False,
        max_cooccurrence_entities: int = 12,
    ) -> KnowledgeGraph:
        """Build complete knowledge graph from transcript data."""
        self.build_speaker_map(data)
        self.add_speaker_nodes()
        self.add_legislation_nodes(data)

        transcripts = data.get("transcripts", [])
        print(f"Processing {len(transcripts)} transcript entries...")

        if granularity not in {"entry", "paragraph"}:
            raise ValueError("granularity must be 'entry' or 'paragraph'")

        if granularity == "entry":
            for i, transcript in enumerate(transcripts, 1):
                if i % batch_size == 0:
                    print(f"Processed {i}/{len(transcripts)} entries...")

                entities = self.extract_entities(transcript)
                if entities:
                    relationships = self.extract_relationships(
                        entities,
                        transcript.get("text", ""),
                        transcript.get("speaker_id", ""),
                        transcript.get("start", ""),
                        include_numeric_entities=include_numeric_entities_in_relations,
                        max_entities=max_entities_per_prompt,
                    )
                    self.relationships.extend(relationships)
        else:
            paragraphs = group_transcripts_into_paragraphs(
                youtube_video_id, transcripts
            )
            print(
                f"Building relationships at paragraph granularity: {len(paragraphs)} paragraphs"
            )

            paragraphs_with_entities = 0
            paragraphs_with_relation_candidates = 0
            llm_attempts = 0
            llm_edges_added = 0
            speaker_edges_added = 0
            cooccurrence_edges_added = 0

            seen_edges: set[tuple[str, str, str, str]] = set()

            for i, paragraph in enumerate(paragraphs, 1):
                if i % batch_size == 0:
                    print(f"Processed {i}/{len(paragraphs)} paragraphs...")

                text = paragraph.get_text()
                speaker_id = paragraph.speaker_id
                timestamp = paragraph.start_timestamp

                entities = self.extract_entities_from_text(text, speaker_id, timestamp)
                if entities:
                    paragraphs_with_entities += 1

                    # Deterministic edges: Speaker -> Entity mentions.
                    if add_speaker_mention_edges:
                        speaker_node_id = f"speaker_{speaker_id}"
                        unique_entity_ids = {eid for eid, _, _ in entities}
                        for ent_id in unique_entity_ids:
                            key = (speaker_node_id, ent_id, "MENTIONS", timestamp)
                            if key in seen_edges:
                                continue
                            seen_edges.add(key)
                            self.relationships.append(
                                Relationship(
                                    source_id=speaker_node_id,
                                    target_id=ent_id,
                                    relation="MENTIONS",
                                    context=text[:200] + "..."
                                    if len(text) > 200
                                    else text,
                                    speaker_id=speaker_id,
                                    timestamp=timestamp,
                                )
                            )
                            speaker_edges_added += 1

                    # Optional deterministic edges: co-occurrence inside paragraph.
                    if add_cooccurrence_edges:
                        # Reuse the same filtering rules as relationship prompts.
                        unique_by_id: dict[str, tuple[str, str]] = {}
                        for eid, etext, etype in entities:
                            if eid not in unique_by_id:
                                unique_by_id[eid] = (etext, etype)

                        items = [
                            (eid, etype)
                            for eid, (_, etype) in unique_by_id.items()
                            if include_numeric_entities_in_relations
                            or etype in self.DEFAULT_RELATION_ENTITY_TYPES
                        ]
                        items = items[:max_cooccurrence_entities]
                        ids = [eid for eid, _ in items]
                        for i1 in range(len(ids)):
                            for i2 in range(i1 + 1, len(ids)):
                                a = ids[i1]
                                b = ids[i2]
                                key = (a, b, "CO_OCCURS_WITH", timestamp)
                                if key in seen_edges:
                                    continue
                                seen_edges.add(key)
                                self.relationships.append(
                                    Relationship(
                                        source_id=a,
                                        target_id=b,
                                        relation="CO_OCCURS_WITH",
                                        context=text[:200] + "..."
                                        if len(text) > 200
                                        else text,
                                        speaker_id=speaker_id,
                                        timestamp=timestamp,
                                    )
                                )
                                cooccurrence_edges_added += 1

                    relationships = self.extract_relationships(
                        entities,
                        text,
                        speaker_id,
                        timestamp,
                        include_numeric_entities=include_numeric_entities_in_relations,
                        max_entities=max_entities_per_prompt,
                    )
                    # We consider an LLM attempt only if we had enough eligible entities.
                    # extract_relationships() returns [] if fewer than 2 eligible.
                    if relationships:
                        llm_edges_added += len(relationships)
                    if relationships or True:
                        # We don't have an explicit hook for "called but returned nothing".
                        # Approximate: if there are >= 2 eligible entities, extract_relationships
                        # will call Gemini. We infer that here by re-applying the same filtering.
                        unique_ids: dict[str, dict[str, str]] = {}
                        for eid, etext, etype in entities:
                            if eid not in unique_ids:
                                unique_ids[eid] = {
                                    "id": eid,
                                    "text": etext,
                                    "type": etype,
                                }
                        eligible = list(unique_ids.values())
                        if not include_numeric_entities_in_relations:
                            eligible = [
                                e
                                for e in eligible
                                if e.get("type") in self.DEFAULT_RELATION_ENTITY_TYPES
                            ]
                        eligible = eligible[:max_entities_per_prompt]
                        if len(eligible) >= 2:
                            paragraphs_with_relation_candidates += 1
                            llm_attempts += 1
                    self.relationships.extend(relationships)

            print(
                "Relationship extraction stats: "
                f"paragraphs_with_entities={paragraphs_with_entities} "
                f"paragraphs_with_relation_candidates={paragraphs_with_relation_candidates} "
                f"llm_attempts={llm_attempts} llm_edges_added={llm_edges_added} "
                f"speaker_edges_added={speaker_edges_added} "
                f"cooccurrence_edges_added={cooccurrence_edges_added}"
            )

        print(
            f"Extraction complete. {len(self.entities)} entities, {len(self.relationships)} relationships."
        )

        upload_date_str = data.get("video_metadata", {}).get("upload_date", "")
        if upload_date_str and len(upload_date_str) >= 8:
            try:
                anchor_date = datetime.strptime(upload_date_str[:8], "%Y%m%d")
                print(f"Using anchor date: {anchor_date.strftime('%Y-%m-%d')}")
            except ValueError as e:
                print(f"Error: Could not parse upload date '{upload_date_str}': {e}")
                print(
                    "Error: Valid anchor date required for relative date normalization"
                )
                raise
        else:
            print(
                "Error: No valid upload date found in video metadata. "
                "Valid anchor date required for relative date normalization."
            )
            raise ValueError("No valid anchor date found in video metadata")

        print("Normalizing relative dates...")
        self.normalize_relative_dates(anchor_date)

        return self._create_knowledge_graph()

    def _create_knowledge_graph(self) -> KnowledgeGraph:
        """Create KnowledgeGraph object from extracted data."""
        nodes = []
        for entity in self.entities.values():
            node_data = {
                "id": entity.id,
                "text": entity.text,
                "type": entity.type,
                "speaker_context": entity.speaker_context,
                "timestamps": entity.timestamps,
                "count": entity.count,
            }

            if entity.type == "DATE":
                if entity.is_relative and entity.resolved_date:
                    node_data["resolved_date"] = entity.resolved_date
                node_data["is_relative"] = entity.is_relative

            nodes.append(node_data)

        edges = []
        for rel in self.relationships:
            edges.append(
                {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "relationship": rel.relation,
                    "context": rel.context,
                    "speaker_id": rel.speaker_id,
                    "timestamp": rel.timestamp,
                }
            )

        entity_types = defaultdict(int)
        for entity in self.entities.values():
            entity_types[entity.type] += 1

        statistics = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": dict(entity_types),
            "relation_types": self._count_relation_types(),
        }

        return KnowledgeGraph(nodes=nodes, edges=edges, statistics=statistics)

    def _count_relation_types(self) -> dict[str, int]:
        """Count occurrences of each relation type."""
        counts = defaultdict(int)
        for rel in self.relationships:
            counts[rel.relation] += 1
        return dict(counts)

    def export_json(self, kg: KnowledgeGraph, filepath: str):
        """Export knowledge graph to JSON file."""
        output = {"nodes": kg.nodes, "edges": kg.edges, "statistics": kg.statistics}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Knowledge graph exported to {filepath}")

    def export_networkx_html(self, kg: KnowledgeGraph, filepath: str):
        """Export knowledge graph as interactive HTML visualization."""
        G = nx.DiGraph()

        color_map = {
            "PERSON": "#FF6B6B",
            "ORG": "#4ECDC4",
            "GPE": "#45B7D1",
            "LAW": "#96CEB4",
            "LEGISLATION": "#FFA07A",
            "SPEAKER": "#FFD93D",
            "DATE": "#A8E6CF",
            "MONEY": "#FF8B94",
            "PERCENT": "#C7CEEA",
            "CARDINAL": "#B8E0D2",
            "ORDINAL": "#D4A5A9",
            "NORP": "#FFB5A7",
        }

        for node in kg.nodes:
            node_type = node["type"]
            color = color_map.get(node_type, "#CCCCCC")
            size = 20 + min(node["count"] * 5, 30)

            title = f"Type: {node['type']}\nMentions: {node['count']}\nContexts: {len(node['speaker_context'])}"
            if (
                node_type == "DATE"
                and node.get("is_relative")
                and node.get("resolved_date")
            ):
                title += f"\nResolved: {node['resolved_date']}"

            G.add_node(
                node["id"],
                label=f"{node['text']}\n({node['type']})",
                title=title,
                color=color,
                size=size,
                group=node_type,
            )

        for edge in kg.edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                title=f"{edge['relationship']}\nSpeaker: {edge['speaker_id']}\nTime: {edge['timestamp']}",
                label=edge["relationship"],
                color="#888888",
            )

        net = Network(
            height="900px", width="100%", notebook=False, cdn_resources="remote"
        )
        net.from_nx(G)

        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          },
          "nodes": {
            "font": {"size": 12},
            "borderWidth": 2
          },
          "edges": {
            "font": {"size": 10, "align": "middle"},
            "smooth": {"type": "cubicBezier"}
          }
        }
        """)

        net.save_graph(filepath)
        print(f"Interactive visualization exported to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Build knowledge graph from parliamentary transcripts"
    )
    parser.add_argument(
        "--input-file",
        default="transcription_output.json",
        help="Input transcript JSON file",
    )
    parser.add_argument(
        "--output-json",
        default="knowledge_graph.json",
        help="Output JSON file for knowledge graph",
    )
    parser.add_argument(
        "--output-html",
        default="knowledge_graph.html",
        help="Output HTML file for visualization",
    )
    parser.add_argument(
        "--gemini-api-key", help="Gemini API key (default: use GOOGLE_API_KEY env var)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of transcript entries to process before progress update",
    )
    parser.add_argument(
        "--granularity",
        choices=["entry", "paragraph"],
        default="paragraph",
        help="Relationship extraction granularity (paragraph reduces LLM calls)",
    )
    parser.add_argument(
        "--youtube-video-id",
        default="unknown_video",
        help="Video id used for paragraph grouping (only for paragraph granularity)",
    )
    parser.add_argument(
        "--include-numeric-entities-in-relations",
        action="store_true",
        help="Include DATE/TIME/CARDINAL/etc entities in relationship prompts",
    )
    parser.add_argument(
        "--max-entities-per-prompt",
        type=int,
        default=20,
        help="Max unique entities to include in each relationship prompt",
    )
    parser.add_argument(
        "--no-speaker-mention-edges",
        action="store_true",
        help="Disable deterministic Speaker->Entity MENTIONS edges",
    )
    parser.add_argument(
        "--add-cooccurrence-edges",
        action="store_true",
        help="Add deterministic CO_OCCURS_WITH edges between entities in each paragraph",
    )
    parser.add_argument(
        "--max-cooccurrence-entities",
        type=int,
        default=12,
        help="Max entities per paragraph to use for CO_OCCURS_WITH edges",
    )

    args = parser.parse_args()

    api_key = args.gemini_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(
            "Error: Please set GOOGLE_API_KEY environment variable or provide --gemini-api-key"
        )
        return 1

    print(f"Loading transcript data from {args.input_file}...")
    builder = KnowledgeGraphBuilder(api_key)

    data = builder.load_transcript_data(args.input_file)

    print("Building knowledge graph...")
    kg = builder.build_knowledge_graph(
        data,
        batch_size=args.batch_size,
        granularity=args.granularity,
        youtube_video_id=args.youtube_video_id,
        include_numeric_entities_in_relations=args.include_numeric_entities_in_relations,
        max_entities_per_prompt=args.max_entities_per_prompt,
        add_speaker_mention_edges=not args.no_speaker_mention_edges,
        add_cooccurrence_edges=args.add_cooccurrence_edges,
        max_cooccurrence_entities=args.max_cooccurrence_entities,
    )

    print(f"Exporting to JSON: {args.output_json}")
    builder.export_json(kg, args.output_json)

    print(f"Exporting to HTML: {args.output_html}")
    builder.export_networkx_html(kg, args.output_html)

    print("\nStatistics:")
    print(f"  Total nodes: {kg.statistics['total_nodes']}")
    print(f"  Total edges: {kg.statistics['total_edges']}")
    print("\nEntity types:")
    for entity_type, count in kg.statistics["entity_types"].items():
        print(f"  {entity_type}: {count}")
    print("\nRelation types:")
    for rel_type, count in kg.statistics["relation_types"].items():
        print(f"  {rel_type}: {count}")

    return 0


if __name__ == "__main__":
    exit(main())
