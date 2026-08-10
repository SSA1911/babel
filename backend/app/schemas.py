"""Structured JSON message shapes exchanged with the client over WebSockets."""

from typing import Literal

from pydantic import BaseModel


class TranscriptMessage(BaseModel):
    type: Literal["transcript"] = "transcript"
    segment_id: int
    data: str
    is_final: bool
    language: str  # active Deepgram source language code at the time this segment arrived


class TranslationMessage(BaseModel):
    type: Literal["translation"] = "translation"
    segment_id: int
    data: str
    final: bool


class StatusMessage(BaseModel):
    """data is a short human-readable status string in the common case.

    /ws/stream also sends two other shapes on this same "status" type,
    neither modeled here since these are sent as raw dicts (see main.py's
    _safe_send_json), not validated against this class:
      - on connect: {"type": "status", "data": "connected",
        "multi_mode": bool, "language": "<code>"} -- multi_mode tells the
        client whether this session auto-switches languages via Deepgram's
        native code-switching, or needs a manual language picker.
      - on a manual switch: {"type": "status", "data": "language_switched",
        "language": "<code>"}, sent after a client "switch_language"
        control message succeeds (see main.py's switch_to_language).
    """

    type: Literal["status"] = "status"
    data: str


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    data: str


class SpeakerTurnMessage(BaseModel):
    """Server -> client, sent on /ws/diarize as diart resolves each window."""

    type: Literal["speaker_turn"] = "speaker_turn"
    speaker: str
    start: float
    end: float


class TranslateRequest(BaseModel):
    """Client -> server, sent on /ws/translate for standalone text translation
    (used when the client already has transcribed text, e.g. from on-device
    STT, and just needs a cloud translation fallback)."""

    segment_id: int
    text: str
    target_lang: str


class SwitchLanguageRequest(BaseModel):
    """Client -> server, sent as a text frame on /ws/stream to manually
    switch the active source language. Documented here but not validated
    against this class in main.py (handle_control_message parses it as a
    plain dict) -- malformed/unknown values are simply ignored rather than
    erroring, since a manual switch is a UI action, not a request needing
    error feedback the way /ws/translate's does."""

    type: Literal["switch_language"] = "switch_language"
    language: str  # must be one of the session's candidate source languages
