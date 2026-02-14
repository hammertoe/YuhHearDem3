from lib.chat_agent_v2 import KGChatAgentV2, _extract_answer_citation_ids, _merge_cite_utterance_ids


def test_extract_answer_citation_ids_should_parse_supported_link_formats() -> None:
    answer = (
        "Teachers issue [1](#src:utt_1). "
        "Mutual recognition [2](source:utt_2). "
        "Full link [3](https://example.com/chat#src:utt_3). "
        "External [x](https://example.com)."
    )

    got = _extract_answer_citation_ids(answer)

    assert got == ["utt_1", "utt_2", "utt_3"]


def test_extract_answer_citation_ids_should_parse_grouped_and_spaced_links() -> None:
    answer = "Grouped [cite] (#src:utt_1, #src:utt_2). Source format [x](source:utt_3)."

    got = _extract_answer_citation_ids(answer)

    assert got == ["utt_1", "utt_2", "utt_3"]


def test_extract_answer_citation_ids_should_parse_malformed_closing_bracket_links() -> None:
    answer = "Broken [cite] (#src:utt_1]. Also [2](#src:utt_2]."

    got = _extract_answer_citation_ids(answer)

    assert got == ["utt_1", "utt_2"]


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


def test_merge_cite_utterance_ids_should_expand_grouped_src_links() -> None:
    retrieval = {
        "citations": [
            {"utterance_id": "utt_1"},
            {"utterance_id": "utt_2"},
        ]
    }

    got = _merge_cite_utterance_ids(
        answer="Grouped [cite](#src:utt_1, #src:utt_2)",
        cite_utterance_ids=[],
        retrieval=retrieval,
    )

    assert got == ["utt_1", "utt_2"]


def test_merge_cite_utterance_ids_should_resolve_utt_seconds_to_unique_known_id() -> None:
    retrieval = {
        "citations": [
            {"utterance_id": "AEOFDga2dh8:10848"},
            {"utterance_id": "otherVid:220"},
        ]
    }

    got = _merge_cite_utterance_ids(
        answer="Education Bill [cite](#src:utt_10848)",
        cite_utterance_ids=[],
        retrieval=retrieval,
    )

    assert got == ["AEOFDga2dh8:10848"]


def test_merge_cite_utterance_ids_should_resolve_unique_seconds_with_suffix_ids() -> None:
    retrieval = {
        "citations": [
            {"utterance_id": "AEOFDga2dh8:10848_2"},
            {"utterance_id": "otherVid:220"},
        ]
    }

    got = _merge_cite_utterance_ids(
        answer="Education Bill [cite](#src:utt_10848)",
        cite_utterance_ids=[],
        retrieval=retrieval,
    )

    assert got == ["AEOFDga2dh8:10848_2"]


def test_merge_cite_utterance_ids_should_keep_well_formed_unknown_ids_for_db_fallback() -> None:
    retrieval = {
        "citations": [
            {"utterance_id": "Syxyah7QIaM:3383"},
        ]
    }

    got = _merge_cite_utterance_ids(
        answer="Known [1](#src:Syxyah7QIaM:3383) and fallback [2](#src:Q1VXHDpBeAg:1472)",
        cite_utterance_ids=["Syxyah7QIaM:3383"],
        retrieval=retrieval,
    )

    assert got == ["Syxyah7QIaM:3383", "Q1VXHDpBeAg:1472"]


class _FakePostgresForFetch:
    def execute_query(self, _sql: str, _params=None):
        return [
            (
                "Q1VXHDpBeAg:1472",
                "Q1VXHDpBeAg",
                1472,
                "24:32",
                "But I will be very specific...",
                "r_r_straughn_1",
                "Mr. Ryan Straughn",
                "Minister in the Ministry of Finance",
                "The Honourable The House - Tuesday 9th December, 2025 - Part 1",
                "2025-12-09",
            )
        ]


def test_fetch_source_by_id_should_prefer_enriched_speaker_name_and_title() -> None:
    agent = KGChatAgentV2(
        postgres_client=_FakePostgresForFetch(),
        embedding_client=object(),
        client=object(),
    )

    source = agent._fetch_source_by_id("Q1VXHDpBeAg:1472")

    assert source is not None
    assert source.speaker_name == "Mr. Ryan Straughn"
    assert source.speaker_title == "Minister in the Ministry of Finance"
