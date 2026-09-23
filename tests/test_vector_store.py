import asyncio
import math

import pytest

from core.vector_store import VectorStore


def run(coroutine: object) -> object:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_add_and_search_returns_highest_similarity_first() -> None:
    store = VectorStore()
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    metadata = [
        {"document_id": "doc-1", "text": "first"},
        {"document_id": "doc-2", "text": "second"},
    ]

    ids = run(store.add_embeddings(embeddings, metadata))
    results = run(store.search([0.9, 0.1], k=1))

    assert isinstance(ids, list)
    assert len(ids) == 2
    assert len(results) == 1
    assert results[0]["id"] == ids[0]
    assert results[0]["metadata"]["text"] == "first"
    assert math.isclose(results[0]["score"], 0.9 / math.sqrt(0.82))


def test_add_embeddings_requires_matching_lengths() -> None:
    store = VectorStore()

    with pytest.raises(ValueError, match="same number"):
        run(store.add_embeddings([[1.0]], []))


@pytest.mark.parametrize(
    "embedding",
    [[], [math.nan], [math.inf], [-math.inf]],
)
def test_invalid_embeddings_are_rejected(embedding: list[float]) -> None:
    store = VectorStore()

    with pytest.raises(ValueError):
        run(store.add_embeddings([embedding], [{"document_id": "doc-1"}]))


def test_delete_document_removes_only_matching_chunks() -> None:
    store = VectorStore()
    run(
        store.add_embeddings(
            [[1.0], [2.0]],
            [
                {"document_id": "doc-1", "text": "remove"},
                {"document_id": "doc-2", "text": "keep"},
            ],
        )
    )

    run(store.delete_document("doc-1"))
    results = run(store.search([1.0]))

    assert len(results) == 1
    assert results[0]["metadata"]["document_id"] == "doc-2"
