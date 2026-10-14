"""Structured JSON message shapes exchanged with the client over WebSockets.

Deliberately minimal: only the request the server actually validates live.
Outgoing message shapes (status/transcript/translation/error) are produced
as plain dicts in main.py -- duplicating them here is how the previous
version of this file drifted from reality; don't resurrect that.
"""

from typing import Optional

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    """Client -> server, sent on /ws/translate for standalone text translation
    (used when a client already has transcribed text and just needs a cloud
    translation)."""

    segment_id: int
    text: str = Field(max_length=5000)
    target_lang: str = Field(max_length=64)
    # Source language of `text`, when the client knows it. Feeds pick_model's
    # coverage check; absent means "unknown" (all([]) -> fast model, which
    # covers every source language Babel supports anyway).
    source_lang: Optional[str] = Field(default=None, max_length=64)
