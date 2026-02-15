from lib.bills.excerpt_chunker import chunk_bill_pages


def test_chunk_bill_pages_should_preserve_page_numbers() -> None:
    pages = [
        {"page_number": 2, "text": "Water policy clause. " * 80},
        {"page_number": 3, "text": "Sewage and treatment systems. " * 80},
    ]

    chunks = chunk_bill_pages("bill_water", pages, chunk_size=300, overlap=50)

    assert len(chunks) >= 2
    assert all(c.page_number in {2, 3} for c in chunks)
    assert chunks[0].chunk_index == 0
    assert sorted(c.chunk_index for c in chunks) == list(range(len(chunks)))
