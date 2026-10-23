"""Claude access for the API: streaming chat text and schema-validated JSON.

Routers depend on the small :class:`LLM` protocol so tests (and the offline demo mode)
can substitute their own implementation via ``app.state.llm``.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Protocol, TypeVar

from pydantic import BaseModel

from ..settings import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Server-side refusal fallback: if a safety classifier declines a request, the API
# re-runs it on Anthropic's recommended fallback model instead of returning a refusal.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Claude could not produce a usable answer (refusal, network, or API error)."""


class LLM(Protocol):
    @property
    def available(self) -> bool: ...

    def stream_text(
        self, *, system: str, messages: list[dict], max_tokens: int = 2048, effort: str = "low"
    ) -> AsyncIterator[str]: ...

    async def structured(
        self, *, system: str, messages: list[dict], schema: type[T], max_tokens: int = 4096, effort: str = "medium"
    ) -> T: ...

    async def aclose(self) -> None: ...


class UnavailableLLM:
    """Stand-in when no Claude credentials are configured; callers check ``available``."""

    available = False

    async def stream_text(self, **_: object) -> AsyncIterator[str]:
        raise LLMError("Claude is not configured on this server.")
        yield ""  # pragma: no cover  (makes this an async generator)

    async def structured(self, **_: object):
        raise LLMError("Claude is not configured on this server.")

    async def aclose(self) -> None:
        return None


class AnthropicLLM:
    available = True

    def __init__(self, settings: Settings):
        import anthropic

        self._anthropic = anthropic
        key = settings.anthropic_key()
        self.client = anthropic.AsyncAnthropic(api_key=key) if key else anthropic.AsyncAnthropic()
        self.model = settings.llm_model

    async def stream_text(
        self, *, system: str, messages: list[dict], max_tokens: int = 2048, effort: str = "low"
    ) -> AsyncIterator[str]:
        anthropic = self._anthropic
        try:
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": effort},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach Claude. Please try again.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude is busy right now. Please try again in a moment.") from exc
        except anthropic.APIStatusError as exc:
            log.warning("Claude API error %s (request %s)", exc.status_code, getattr(exc, "request_id", None))
            raise LLMError("Claude had a problem answering. Please try again.") from exc
        if final.stop_reason == "refusal":
            raise LLMError("Claude couldn’t answer that one. Try asking another way.")

    async def structured(
        self, *, system: str, messages: list[dict], schema: type[T], max_tokens: int = 4096, effort: str = "medium"
    ) -> T:
        anthropic = self._anthropic
        try:
            response = await self.client.beta.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_format=schema,
                output_config={"effort": effort},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach Claude. Please try again.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude is busy right now. Please try again in a moment.") from exc
        except anthropic.APIStatusError as exc:
            log.warning("Claude API error %s (request %s)", exc.status_code, getattr(exc, "request_id", None))
            raise LLMError("Claude had a problem answering. Please try again.") from exc
        if response.stop_reason == "refusal":
            raise LLMError("Claude couldn’t help with that request.")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError("Claude’s answer was incomplete. Please try again.")
        return response.parsed_output

    async def aclose(self) -> None:
        await self.client.close()


def build_llm(settings: Settings) -> LLM:
    if not settings.llm_configured:
        log.info("Claude is not configured; clarify/converse features use offline fallbacks")
        return UnavailableLLM()
    return AnthropicLLM(settings)
