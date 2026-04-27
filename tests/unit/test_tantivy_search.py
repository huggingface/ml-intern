from agent.search import TantivyTextIndex, chunk_code, chunk_markdown


def test_tantivy_text_index_ranks_field_boosted_hits():
    index = TantivyTextIndex(
        text_fields=["title", "content"],
        stored_fields=["url"],
        field_boosts={"title": 3.0, "content": 1.0},
    )
    index.add_documents(
        [
            {
                "title": "SFTTrainer dataset_text_field",
                "content": "Configuration reference for supervised fine tuning.",
                "url": "https://example.test/sft",
            },
            {
                "title": "Generic training loop",
                "content": "dataset_text_field appears in the body only.",
                "url": "https://example.test/generic",
            },
        ]
    )

    hits, errors = index.search("dataset_text_field", limit=2)

    assert errors == []
    assert [hit.fields["url"] for hit in hits] == [
        "https://example.test/sft",
        "https://example.test/generic",
    ]


def test_chunk_markdown_preserves_heading_and_line_range():
    content = "# Intro\n\nStart here\n\n## Packing\n\nUse packing with SFTConfig."

    chunks = chunk_markdown(content)

    assert chunks[-1].title == "Packing"
    assert chunks[-1].line_start == 5
    assert chunks[-1].line_end == 7
    assert "SFTConfig" in chunks[-1].text


def test_chunk_code_uses_overlapping_line_windows():
    content = "\n".join(f"line_{i}" for i in range(1, 121))

    chunks = chunk_code(content, window=50, overlap=10)

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (1, 50),
        (41, 90),
        (81, 120),
    ]
