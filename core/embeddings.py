import asyncio

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate local text embeddings with a Hugging Face model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Load the model only when it is first needed."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    def _embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")

        embedding = self._get_model().encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty values")

        if not texts:
            return []

        embeddings = self._get_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for one text value."""
        return await asyncio.to_thread(self._embed_text, text)

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple text values."""
        return await asyncio.to_thread(self._embed_batch, texts)