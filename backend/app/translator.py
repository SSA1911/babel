"""Streaming translation of finalized transcript segments via Groq."""

import logging
from typing import AsyncIterator, Optional

from groq import AsyncGroq

from .config import settings

logger = logging.getLogger("babel.translator")

_client = AsyncGroq(api_key=settings.groq_api_key)

# Languages llama-3.3 officially supports. Everything else in Babel's source
# list -- Arabic, Tagalog, Tamil, Chinese, Japanese, Korean, Turkish -- falls
# outside it, which is what the quality model exists for.
FAST_MODEL_LANGUAGES = {"en", "de", "fr", "it", "pt", "hi", "es", "th"}
FAST_MODEL_LANGUAGE_NAMES = {
    "english", "german", "french", "italian", "portuguese", "hindi", "spanish", "thai",
}

# Kept short on purpose: every token here is prefill on every single segment,
# paid again for each utterance in a live conversation.
SYSTEM_PROMPT = (
    "You are a live interpreter. Translate the user's message into {target_language}. "
    "Output only the translation - no notes, quotes, or explanation. "
    "If it is already in {target_language}, repeat it unchanged. "
    "Keep the speaker's tone and register."
)

# Used when two connections heard the same audio differently, which is the
# signature of code-switched speech: each single-language model transcribes
# its own language correctly and mangles the other into phonetic nonsense.
# Neither transcript is right on its own; together they usually determine what
# was actually said.
FUSION_SYSTEM_PROMPT = (
    "You are a live interpreter. Two speech recognizers heard the same sentence, "
    "each tuned to a different language, and the speaker may have mixed both "
    "languages in one sentence. Each recognizer transcribes its own language "
    "correctly and garbles the other into phonetic nonsense. Work out what was "
    "actually said, then translate it into {target_language}. "
    "Output only the translation - no notes, quotes, or explanation. "
    "Keep the speaker's tone and register."
)


def pick_model(source_languages: list[str], target_language: str, mode: Optional[str] = None) -> str:
    """Chooses a Groq model for a session.

    "auto" uses the fast (non-reasoning) model only when every language in
    play is one it officially supports, because that model's coverage stops
    well short of Babel's source list. Anything involving Arabic, Tagalog,
    Tamil or CJK gets the quality model instead -- it is slower to first
    token, but the fast model's output in those languages isn't worth having.
    """
    mode = (mode or settings.groq_model_mode or "auto").lower()
    if mode == "fast":
        return settings.groq_model_fast
    if mode == "quality":
        return settings.groq_model_quality

    covered = all(code in FAST_MODEL_LANGUAGES for code in source_languages)
    covered = covered and target_language.strip().lower() in FAST_MODEL_LANGUAGE_NAMES
    return settings.groq_model_fast if covered else settings.groq_model_quality


def _is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model


async def stream_translation(
    text: str,
    target_language: str,
    model: Optional[str] = None,
    alternate_text: Optional[str] = None,
) -> AsyncIterator[str]:
    """Yields translated text deltas as they arrive from Groq.

    `alternate_text` is a competing transcription of the same audio from
    another language's recognizer. When present, both are sent so the model
    can reconstruct code-switched speech (see FUSION_SYSTEM_PROMPT).
    """
    model = model or pick_model([], target_language)

    if alternate_text:
        system_prompt = FUSION_SYSTEM_PROMPT.format(target_language=target_language)
        user_content = f"Recognizer A: {text}\nRecognizer B: {alternate_text}"
    else:
        system_prompt = SYSTEM_PROMPT.format(target_language=target_language)
        user_content = text

    request: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": settings.translation_max_tokens,
        "stream": True,
    }
    if _is_reasoning_model(model):
        # gpt-oss can't disable reasoning outright (Groq offers only
        # low/medium/high; "none" is Qwen-only), and it defaults to medium.
        # Translation needs none of it, and every reasoning token is dead
        # time before the first word reaches the screen.
        request["reasoning_effort"] = "low"
        request["reasoning_format"] = "hidden"

    stream = await _client.chat.completions.create(**request)

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # Reasoning models stream their chain of thought on a separate field.
        # The previous version read only `.content`, so during the entire
        # reasoning phase it saw empty deltas and the UI sat on "translating…"
        # with no indication anything was happening. Skip it explicitly.
        if getattr(delta, "reasoning", None):
            continue
        if delta.content:
            yield delta.content
