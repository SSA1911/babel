"""Environment-driven configuration for the Babel backend."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved from this file rather than left as a bare ".env", which pydantic
# reads relative to the current working directory -- that made the server
# bootable only from backend/, and fail with a confusing "field required"
# error for the API keys from anywhere else.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    deepgram_api_key: str
    groq_api_key: str

    # ---- Translation ---------------------------------------------------
    # "auto" picks per session from the languages actually in play (see
    # translator.pick_model): the fast model when every language is one
    # llama-3.3 officially supports, the quality model otherwise. "fast" and
    # "quality" pin one regardless. The client can override per session.
    groq_model_mode: str = "auto"
    # Non-reasoning, ~280 tok/s, no reasoning preamble before the first
    # visible token. Officially covers en/de/fr/it/pt/hi/es/th only.
    groq_model_fast: str = "llama-3.3-70b-versatile"
    # Reasoning model, but the only option here with credible quality on
    # Arabic/Tamil/Tagalog/CJK (MMMLU: 82.7 Arabic, 82.9 Korean, 81.3 avg).
    # Always sent with reasoning_effort="low" -- see translator.py.
    groq_model_quality: str = "openai/gpt-oss-120b"
    target_language: str = "English"
    # Ceiling on a translation response. Segments are single utterances, so
    # this only ever truncates a runaway generation, never real output.
    translation_max_tokens: int = 512
    # How many segments may be translated concurrently. Emission stays in
    # spoken order regardless (see main.py's ordered emitter) -- this only
    # stops segment N+1 from waiting on N's network round trip.
    translation_concurrency: int = 3

    # ---- Speech-to-text ------------------------------------------------
    deepgram_stt_model: str = "nova-3"
    source_language: str = "en"

    # Silence (ms) before Deepgram finalizes a segment. This is now the
    # single biggest contributor to perceived lag -- translation itself runs
    # in ~50-100ms, so whatever is set here is most of the wait.
    #
    # Deepgram recommends 100 for code-switching, but at 100 a single spoken
    # sentence reliably splits into three separate bubbles, and each fragment
    # then gets translated without the context of the rest, which reads
    # worse. 200 keeps sentences intact while still cutting ~100ms off the
    # previous 300. Lower it if you want maximum responsiveness over
    # readability.
    endpointing_ms: int = 200
    # Safety net for utterances endpointing misses entirely.
    utterance_end_ms: int = 1000

    # A session opens one Deepgram connection per language it can't
    # code-switch natively (see stt_session.plan_connections). The baseline
    # is what a typical session costs; the ceiling is a runaway guard, not a
    # target -- selecting 3 languages legitimately opens 3 connections.
    stt_connection_baseline: int = 2
    stt_connection_ceiling: int = 4
    # How long to hold a finalized segment waiting for competing
    # transcriptions of the same audio from the other connections.
    arbitration_grace_ms: int = 250
    # Two hypotheses within this confidence margin are treated as genuine
    # ambiguity -- the signal for code-switched speech -- and both are sent
    # to the translator to reconstruct. Set to 0 to disable fusion.
    fusion_margin: float = 0.15

    # Comma-separated list, or "*" for any origin (fine for local dev).
    cors_origins: str = "*"

    # extra="ignore" because pydantic-settings defaults to "forbid", which
    # makes a stale key in someone's .env a hard startup crash. Existing
    # .env files still carry HUGGINGFACE_TOKEN and DIARIZATION_MAX_SPEAKERS
    # from the removed diart diarizer; those are now inert rather than fatal.
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
