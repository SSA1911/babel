"""Streaming translation of finalized transcript segments via Groq."""

import logging
from typing import AsyncIterator

from groq import AsyncGroq

from .config import settings

logger = logging.getLogger("babel.translator")

_client = AsyncGroq(api_key=settings.groq_api_key)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional real-time interpreter embedded in a live captioning "
    "system. Translate the user's message from its detected source language into "
    "{target_language}. Output ONLY the direct translation. Do not include "
    "explanations, notes, pleasantries, disclaimers, or surrounding quotes. "
    "If the message is already in {target_language}, output it unchanged. "
    "Preserve the speaker's tone, register, and intent, and keep the translation "
    "as a single fluent utterance."
)


async def stream_translation(text: str, target_language: str) -> AsyncIterator[str]:
    """Yields translated text tokens/deltas as they arrive from Groq."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(target_language=target_language)

    stream = await _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
