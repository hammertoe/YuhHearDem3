from __future__ import annotations

import re
from dataclasses import dataclass

from build_knowledge_graph import KnowledgeGraphBuilder


@dataclass(frozen=True)
class _Ent:
    text: str
    label_: str


class _Doc:
    def __init__(self, ents: list[_Ent]):
        self.ents = ents


class _NLP:
    def __call__(self, text: str) -> _Doc:
        # Return two entities for the first paragraph, one for the second.
        if "Tax Bill" in text:
            return _Doc([_Ent("Alice", "PERSON"), _Ent("Tax Bill", "LAW")])
        if "inflation" in text:
            return _Doc([_Ent("Bob", "PERSON")])
        return _Doc([])


class _Gemini:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, prompts: list[str]) -> list[str]:
        # One response per prompt.
        out: list[str] = []
        for prompt in prompts:
            self.calls.append(prompt)
            # Return strict JSON with indices.
            if "Tax Bill" in prompt:
                out.append(
                    '[{"source_index": 1, "relation": "DISCUSSES", "target_index": 2, "evidence": "Tax Bill"}]'
                )
            else:
                out.append("[]")
        return out


def test_paragraph_granularity_makes_one_llm_call_per_paragraph() -> None:
    gemini = _Gemini()
    builder = KnowledgeGraphBuilder(
        gemini_api_key="test",
        nlp=_NLP(),
        gemini_model=gemini,
    )

    data = {
        "video_metadata": {"upload_date": "20260106"},
        "speakers": [
            {"speaker_id": "s_1", "name": "Alice"},
            {"speaker_id": "s_2", "name": "Bob"},
        ],
        "legislation": [],
        "transcripts": [
            {
                "speaker_id": "s_1",
                "start": "00:00:01",
                "text": "Alice introduces the Tax Bill.",
            },
            {"speaker_id": "s_1", "start": "00:00:05", "text": "It is urgent."},
            {"speaker_id": "s_2", "start": "00:00:10", "text": "Bob notes inflation."},
        ],
    }

    kg = builder.build_knowledge_graph(
        data,
        batch_size=10,
        granularity="paragraph",
        youtube_video_id="test_video",
    )

    # Only the first paragraph has >=2 entities, so only one LLM call.
    assert len(gemini.calls) == 1

    # Prompt text should include both sentences from the paragraph.
    assert "Alice introduces the Tax Bill." in gemini.calls[0]
    assert "It is urgent." in gemini.calls[0]

    # Should produce at least one edge from the LLM response.
    assert kg.statistics["total_edges"] >= 1
