"""Environment-driven configuration for the Babel backend."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deepgram_api_key: str
    groq_api_key: str
    # Required for /ws/diarize: diart's default pipeline needs two gated
    # models. Accept terms at https://huggingface.co/pyannote/segmentation
    # and https://huggingface.co/pyannote/embedding, generate a token at
    # https://huggingface.co/settings/tokens, then set it here.
    huggingface_token: str = ""

    groq_model: str = "openai/gpt-oss-120b"
    target_language: str = "English"

    # Comma-separated list, or "*" for any origin (fine for local dev).
    cors_origins: str = "*"

    deepgram_stt_model: str = "nova-3"
    # Deepgram's live/streaming API does not support language auto-detection,
    # so a source language must be picked explicitly (see index.html's
    # "Source" dropdown, sent as ?source_lang=).
    source_language: str = "en"

    # diart/pyannote streaming diarization tuning (see app/diarization.py).
    diarization_max_speakers: int = 6

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
