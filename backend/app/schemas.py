"""Structured JSON message shapes exchanged with the client over WebSockets."""

from typing import Literal, Optional

from pydantic import BaseModel


class WordLanguage(BaseModel):
    """One word and the language Deepgram heard it in. Only populated for
    `language=multi` connections, where a single sentence can legitimately
    contain more than one language."""

    word: str
    language: str


class TranscriptMessage(BaseModel):
    type: Literal["transcript"] = "transcript"
    segment_id: int
    data: str
    is_final: bool
    language: str  # language of this segment, decided by arbitration (see stt_session)
    # Speaker index from Deepgram's streaming diarizer, taken from the
    # session's anchor connection so labels stay consistent across languages.
    speaker: Optional[int] = None
    # Present only when one utterance genuinely mixed languages.
    word_languages: Optional[list[WordLanguage]] = None


class TranslationMessage(BaseModel):
    type: Literal["translation"] = "translation"
    segment_id: int
    data: str
    final: bool
    # Wall-clock milliseconds from Deepgram finalizing the segment to the
    # first translated token, and to completion. Sent on the final message
    # so the client can surface real latency instead of guessing at it.
    ms_to_first_token: Optional[int] = None
    ms_total: Optional[int] = None
    # True when this translation was reconstructed from two competing
    # recognizers (code-switched speech) rather than a single transcript.
    fused: bool = False


class StatusMessage(BaseModel):
    """data is a short human-readable status string in the common case.

    /ws/stream also sends a richer shape on this same "status" type, not
    modeled here since it's sent as a raw dict (see main.py's
    _safe_send_json):
      - on connect: {"type": "status", "data": "connected",
        "languages": [<code>, ...],      # auto-detected this session
        "manual_languages": [<code>, ...],  # beyond the connection ceiling
        "connections": int,              # live Deepgram connections
        "diarization": bool,             # speaker labels available
        "model": "<groq model id>"}
      - on a manual switch: {"type": "status", "data": "language_switched",
        "language": "<code>"}, sent after a client "switch_language"
        control message succeeds.
    """

    type: Literal["status"] = "status"
    data: str


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    data: str


class TranslateRequest(BaseModel):
    """Client -> server, sent on /ws/translate for standalone text translation
    (used when a client already has transcribed text and just needs a cloud
    translation)."""

    segment_id: int
    text: str
    target_lang: str


class SwitchLanguageRequest(BaseModel):
    """Client -> server, sent as a text frame on /ws/stream to manually pick
    the active source language.

    Only relevant when a session selected more languages than
    STT_CONNECTION_CEILING allows connections for; in the normal case every
    selected language is transcribed concurrently and detected automatically,
    so there is nothing to switch. Documented here but not validated against
    this class in main.py -- unknown values are ignored rather than erroring,
    since this is a UI action, not a request needing error feedback."""

    type: Literal["switch_language"] = "switch_language"
    language: str
