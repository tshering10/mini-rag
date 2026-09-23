import asyncio
import logging
from typing import Any

from google import genai
from google.genai import errors

logger = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when Gemini cannot generate an answer."""


class LLMService:
    """Generate grounded answers with Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None,
        model_name: str = "gemini-3.6-flash",
        client: Any | None = None,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("GEMINI_API_KEY must be configured")

        self.model_name = model_name
        self._client = client or genai.Client(api_key=api_key)

    async def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate an answer using only the supplied document context."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if not context.strip():
            raise ValueError("context must not be empty")

        prompt = self._build_prompt(question, context)
        response = None
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                )
                break
            except errors.ServerError as error:
                logger.warning(
                    "Gemini server error model=%s status=%s attempt=%d",
                    self.model_name,
                    getattr(error, "code", "unknown"),
                    attempt + 1,
                )
                if attempt == 2:
                    raise LLMServiceError(
                        "Gemini is temporarily unavailable. Please try again later."
                    ) from error
                await asyncio.sleep(2**attempt)
            except errors.APIError as error:
                status_code = getattr(error, "code", None)
                logger.error(
                    "Gemini API error model=%s status=%s message=%s",
                    self.model_name,
                    status_code or "unknown",
                    str(error),
                )
                if status_code in (401, 403):
                    message = (
                        "Gemini authentication failed. Check or rotate "
                        "GEMINI_API_KEY."
                    )
                elif status_code == 404:
                    message = (
                        f"Gemini model '{self.model_name}' was not found "
                        "or is unavailable for this API."
                    )
                elif status_code == 429:
                    message = (
                        "Gemini quota or rate limit reached. "
                        "Please try again later."
                    )
                else:
                    message = "Gemini could not generate an answer for this request."
                raise LLMServiceError(
                    message
                ) from error

        if response is None:
            raise LLMServiceError(
                "Gemini did not return a response. Please try again later."
            )

        answer = response.text

        if not answer or not answer.strip():
            raise LLMServiceError("Gemini returned an empty response")

        return answer.strip()

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            "Answer the question using only the provided context. "
            "If the context does not contain the answer, say that you "
            "do not have enough information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
