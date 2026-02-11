from lib.chat_agent_v2 import _extract_answer_citation_ids, _merge_cite_utterance_ids


def test_extract_answer_citation_ids_should_parse_supported_link_formats() -> None:
    answer = (
        "Teachers issue [1](#src:utt_1). "
        "Mutual recognition [2](source:utt_2). "
        "Full link [3](https://example.com/chat#src:utt_3). "
        "External [x](https://example.com)."
    )

    got = _extract_answer_citation_ids(answer)

    assert got == ["utt_1", "utt_2", "utt_3"]


def test_merge_cite_utterance_ids_should_include_answer_links_and_cite_ids() -> None:
    retrieval = {
        "citations": [
            {"utterance_id": "utt_1"},
            {"utterance_id": "utt_2"},
            {"utterance_id": "utt_3"},
        ]
    }

    got = _merge_cite_utterance_ids(
        answer="Point [1](#src:utt_1) and [2](#src:utt_2)",
        cite_utterance_ids=["utt_3"],
        retrieval=retrieval,
    )

    assert got == ["utt_1", "utt_2", "utt_3"]


def test_merge_cite_utterance_ids_should_filter_unknown_ids_when_retrieval_known() -> None:
    retrieval = {"citations": [{"utterance_id": "utt_1"}]}

    got = _merge_cite_utterance_ids(
        answer="Bad [x](#src:utt_missing)",
        cite_utterance_ids=["utt_1", "utt_missing"],
        retrieval=retrieval,
    )

    assert got == ["utt_1"]


def test_merge_cite_utterance_ids_should_match_known_ids_with_utt_prefix_variants() -> None:
    retrieval = {
        "citations": [
            {"utterance_id": "utt_abc"},
            {"utterance_id": "utt_def"},
        ]
    }

    got = _merge_cite_utterance_ids(
        answer="Point [1](#src:abc)",
        cite_utterance_ids=["def"],
        retrieval=retrieval,
    )

    assert got == ["utt_abc", "utt_def"]
