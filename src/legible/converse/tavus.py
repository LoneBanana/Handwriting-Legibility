"""Tavus CVI client: an optional realistic face and voice for the conversation avatar.

Legible runs Tavus in **echo mode**. The server still writes every word with Claude, so
answers stay short, simple and emotion-tagged and the comprehension check knows exactly
what was said; Tavus only renders the face and voice. The browser joins the Daily room
created here and speaks each answer by sending its ``echo_text`` as an app message.

Checked against the Tavus documentation (docs.tavus.io) on 2026-10-23:

* Create a conversation, https://docs.tavus.io/api-reference/conversations/create-conversation:
  ``POST {api_base}/conversations`` with the ``x-api-key`` header and a JSON body of
  ``face_id`` and ``pal_id``, ``conversation_name``, ``require_auth`` (a private room;
  the response then carries a short-lived ``meeting_token``) and ``properties``
  (``max_call_duration``, ``participant_left_timeout``, ``participant_absent_timeout``,
  ``enable_closed_captions``). The response has ``conversation_id``,
  ``conversation_url`` (a Daily room), ``status`` and ``meeting_token``. Errors carry
  ``error`` (validation) or ``message`` (auth, limits).
* Tavus now calls replicas "faces" and personas "PALs"; ``replica_id`` / ``persona_id``
  and ``/v2/replicas`` / ``/v2/personas`` remain accepted aliases
  (https://docs.tavus.io/sections/conversational-video-interface/pal/what-is-a-pal), so
  ``LEGIBLE_TAVUS_REPLICA_ID`` / ``LEGIBLE_TAVUS_PERSONA_ID`` hold a face id and a PAL id.
* End a conversation, https://docs.tavus.io/api-reference/conversations/end-conversation:
  ``POST {api_base}/conversations/{conversation_id}/end`` with no body.
* Echo mode, https://docs.tavus.io/sections/conversational-video-interface/echo-mode,
  belongs to the PAL, not the conversation. Create the PAL once, e.g.
  ``POST https://tavusapi.com/v2/pals`` with
  ``{"pal_name": "Legible guide", "pipeline_mode": "echo", "default_face_id": "<face id>"}``,
  and set ``LEGIBLE_TAVUS_PERSONA_ID`` to the returned ``pal_id``. Text echoes skip
  Tavus's speech recognition and LLM and go straight to text-to-speech; echo mode does
  not support Raven perception.
* The app message (sent by the browser with Daily's ``call.sendAppMessage(message, "*")``,
  https://docs.tavus.io/sections/conversational-video-interface/interactions-protocols/overview):
  ``{"message_type": "conversation", "event_type": "conversation.echo",
  "conversation_id": ..., "properties": {"modality": "text", "text": echo_text, "done": true}}``.
* Emotion, https://docs.tavus.io/sections/conversational-video-interface/quickstart/emotional-expression:
  Phoenix-4 and Phoenix-4.5 faces read ``<emotion value="excited"/>`` tags written
  before the text they apply to; the values are exactly Legible's vocabulary. There is
  no parameter for expression strength, so ``expression_intensity`` (and so the
  ``converse.expressions`` track's subtlety) only affects the built-in avatar.
* Speech speed, https://docs.tavus.io/sections/conversational-video-interface/pal/tts:
  only on the PAL's TTS layer, either one global ``layers.tts.voice_settings.speed``
  (ElevenLabs 0.7-1.2) or, for Cartesia sonic-3, ``<speed level="0.8">...</speed>`` SSML
  emitted by Tavus's own LLM ("not both"). Neither is documented per conversation or
  for echo text, so ``words_per_minute`` likewise only paces the built-in avatar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx

if TYPE_CHECKING:
    from ..api.settings import Settings

DEFAULT_API_BASE = "https://tavusapi.com/v2"


class TavusError(RuntimeError):
    """Tavus could not be reached or refused the request."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class TavusConversation:
    conversation_id: str
    conversation_url: str
    meeting_token: str | None
    status: str | None = None


class TavusClient:
    """Creates and ends echo-mode conversations for one configured face and PAL.

    Each call opens its own short-lived HTTP client: conversations start and end
    rarely, and nothing then has to be closed when the app shuts down.
    """

    def __init__(
        self,
        api_key: str,
        *,
        face_id: str,
        pal_id: str,
        api_base: str = DEFAULT_API_BASE,
        max_call_seconds: int = 900,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._api_key = api_key
        self.face_id = face_id
        self.pal_id = pal_id
        self.api_base = api_base.rstrip("/")
        self.max_call_seconds = max_call_seconds
        self.timeout = timeout
        self._transport = transport  # tests pass an httpx.MockTransport

    @classmethod
    def from_settings(cls, settings: Settings) -> TavusClient | None:
        """A client when an API key, a face, and an echo-mode PAL are all configured."""
        if not (settings.tavus_configured and settings.tavus_persona_id and settings.tavus_api_key):
            return None
        return cls(
            settings.tavus_api_key.get_secret_value(),
            face_id=settings.tavus_replica_id or "",
            pal_id=settings.tavus_persona_id,
            api_base=settings.tavus_api_base,
            max_call_seconds=settings.tavus_max_call_seconds,
        )

    async def create_conversation(self, name: str) -> TavusConversation:
        body = {
            "face_id": self.face_id,
            "pal_id": self.pal_id,
            "conversation_name": name,
            "require_auth": True,  # only the learner holding the meeting token can join
            "properties": {
                "max_call_duration": self.max_call_seconds,
                "participant_left_timeout": 30,
                "participant_absent_timeout": 120,
                # Legible draws its own captions so they can fade with the listening track.
                "enable_closed_captions": False,
            },
        }
        data = await self._request("POST", "/conversations", json=body)
        conversation_id = data.get("conversation_id")
        conversation_url = data.get("conversation_url")
        if not isinstance(conversation_id, str) or not isinstance(conversation_url, str):
            raise TavusError("Tavus did not return a conversation id and URL.")
        token = data.get("meeting_token")
        status = data.get("status")
        return TavusConversation(
            conversation_id=conversation_id,
            conversation_url=conversation_url,
            meeting_token=token if isinstance(token, str) and token else None,
            status=status if isinstance(status, str) else None,
        )

    async def end_conversation(self, conversation_id: str) -> None:
        await self._request("POST", f"/conversations/{quote(conversation_id, safe='')}/end")

    async def _request(self, method: str, path: str, *, json: dict | None = None) -> dict:
        headers = {"x-api-key": self._api_key, "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                response = await client.request(method, f"{self.api_base}{path}", json=json, headers=headers)
        except httpx.HTTPError as exc:
            raise TavusError(f"Could not reach Tavus: {exc.__class__.__name__}") from exc
        if response.status_code >= 400:
            raise TavusError(
                f"Tavus answered {response.status_code}: {_error_detail(response)}", status_code=response.status_code
            )
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}


def _error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            if isinstance(data.get(key), str):
                return data[key]
    return response.reason_phrase
