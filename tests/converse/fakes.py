"""A scripted stand-in for ``app.state.llm`` (the API's ``LLM`` protocol)."""

from __future__ import annotations

from typing import Any, AsyncIterator

from legible.api.services.llm import LLMError


class FakeLLM:
    """Streams ``chunks`` (an exception in the list is raised at that point) and returns ``check``."""

    available = True

    def __init__(self, chunks: list[Any] | None = None, check: dict | Exception | None = None):
        self.chunks = list(chunks or [])
        self.check = check
        self.stream_calls: list[dict[str, Any]] = []
        self.structured_calls: list[dict[str, Any]] = []

    async def stream_text(
        self, *, system: str, messages: list[dict], max_tokens: int = 2048, effort: str = "low"
    ) -> AsyncIterator[str]:
        self.stream_calls.append({"system": system, "messages": messages, "max_tokens": max_tokens, "effort": effort})
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def structured(
        self, *, system: str, messages: list[dict], schema: type, max_tokens: int = 4096, effort: str = "medium"
    ):
        self.structured_calls.append({"system": system, "messages": messages, "schema": schema, "effort": effort})
        if isinstance(self.check, Exception):
            raise self.check
        if self.check is None:
            raise LLMError("Claude had a problem answering. Please try again.")
        return schema.model_validate(self.check)

    async def aclose(self) -> None:
        return None


def chunked(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]
