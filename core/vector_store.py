import math
from dataclasses import dataclass
from uuid import uuid4
from typing import Any


@dataclass
class StoredEmbedding:
    embedding: list[float]
    metadata: dict[str, Any]


class VectorStore:
    """Store and search embeddings in memory using cosine similarity."""

    def __init__(self) -> None:
        self._items: dict[str, StoredEmbedding] = {}

    async def add_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> list[str]:
        """Store embeddings and their metadata, returning generated IDs."""
        if len(embeddings) != len(metadata):
            raise ValueError(
                "embeddings and metadata must contain the same number of items"
            )

        ids: list[str] = []

        for embedding, item_metadata in zip(embeddings, metadata):
            self._validate_embedding(embedding)

            item_id = str(uuid4())
            self._items[item_id] = StoredEmbedding(
                embedding=list(embedding),
                metadata=dict(item_metadata),
            )
            ids.append(item_id)

        return ids

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-k items ordered by cosine similarity."""
        if k <= 0:
            raise ValueError("k must be greater than zero")

        self._validate_embedding(query_embedding)

        results: list[dict[str, Any]] = []

        for item_id, item in self._items.items():
            if len(item.embedding) != len(query_embedding):
                continue

            score = self._cosine_similarity(query_embedding, item.embedding)

            results.append(
                {
                    "id": item_id,
                    "score": score,
                    "metadata": dict(item.metadata),
                }
            )

        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:k]

    async def delete_document(self, document_id: str) -> None:
        """Delete every stored chunk belonging to a document."""
        ids_to_delete = [
            item_id
            for item_id, item in self._items.items()
            if item.metadata.get("document_id") == document_id
        ]

        for item_id in ids_to_delete:
            del self._items[item_id]

    @staticmethod
    def _validate_embedding(embedding: list[float]) -> None:
        if not embedding:
            raise ValueError("embedding must not be empty")

        if any(not math.isfinite(value) for value in embedding):
            raise ValueError("embedding values must be finite")

    @staticmethod
    def _cosine_similarity(
        first: list[float],
        second: list[float],
    ) -> float:
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))

        if first_norm == 0 or second_norm == 0:
            raise ValueError("embedding vectors must not have zero magnitude")

        dot_product = sum(
            first_value * second_value
            for first_value, second_value in zip(first, second)
        )

        return dot_product / (first_norm * second_norm)