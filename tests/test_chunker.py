import pytest

from core.chunker import TextChunker


def test_empty_text_returns_no_chunks() -> None:
    chunker = TextChunker(chunk_size=10, overlap=2)

    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   ") == []


def test_short_text_returns_one_chunk() -> None:
    chunker = TextChunker(chunk_size=20, overlap=5)

    chunks = chunker.chunk_text("Hello world")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world"
    assert chunks[0]["chunk_id"] == "chunk-0"
    assert chunks[0]["chunk_index"] == 0


def test_long_text_creates_overlapping_chunks() -> None:
    chunker = TextChunker(chunk_size=10, overlap=3)

    chunks = chunker.chunk_text("12345678901234567890")

    assert len(chunks) == 3
    assert chunks[0]["text"] == "1234567890"
    assert chunks[1]["text"] == "8901234567"
    assert chunks[2]["text"] == "567890"


def test_page_number_is_preserved() -> None:
    chunker = TextChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk_text("Some page text", page_number=3)

    assert all(chunk["page_number"] == 3 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_settings_raise_error(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=chunk_size, overlap=overlap)