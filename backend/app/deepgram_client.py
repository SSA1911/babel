"""Thin async wrapper around Deepgram's live transcription WebSocket API."""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

logger = logging.getLogger("babel.deepgram")

DEEPGRAM_LIVE_BASE_URL = "wss://api.deepgram.com/v1/listen"

# Deepgram closes an idle connection after ~10s of no audio. The client
# already streams continuously while listening, so this only matters across
# long silences; a KeepAlive well inside that window costs nothing.
KEEPALIVE_INTERVAL_SECONDS = 5.0


class DeepgramLiveClient:
    """Owns one persistent Deepgram streaming connection for a single language.

    A session may run several of these at once (one per language that
    Deepgram can't code-switch natively) -- see app/stt_session.py.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nova-3",
        language: str = "en",
        diarize: bool = False,
        endpointing_ms: int = 100,
        utterance_end_ms: int = 1000,
    ):
        self._api_key = api_key
        self._model = model
        self._language = language
        self._diarize = diarize
        self._endpointing_ms = endpointing_ms
        self._utterance_end_ms = utterance_end_ms
        self._ws: Optional[ClientConnection] = None
        self._keepalive_task: Optional[asyncio.Task] = None

    @property
    def language(self) -> str:
        return self._language

    @property
    def diarize(self) -> bool:
        return self._diarize

    async def connect(self) -> None:
        params = {
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
            "interim_results": "true",
            "endpointing": str(self._endpointing_ms),
            "utterance_end_ms": str(self._utterance_end_ms),
            "smart_format": "true",
            "model": self._model,
            "language": self._language,
        }
        if self._diarize:
            # NOTE: `diarize=true`, not `diarize_model`. Deepgram's docs mark
            # `diarize` deprecated in favour of `diarize_model`, but that
            # applies to batch only -- `diarize_model` returns 400 on
            # streaming requests (Deepgram changelog, 2026-05-13), so
            # streaming diarization must keep using this parameter.
            params["diarize"] = "true"

        url = f"{DEEPGRAM_LIVE_BASE_URL}?{urlencode(params)}"
        self._ws = await connect(
            url,
            additional_headers={"Authorization": f"Token {self._api_key}"},
            ping_interval=5,
            ping_timeout=20,
            max_size=None,
        )
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name=f"dg_keepalive_{self._language}"
        )

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
                ws = self._ws
                if ws is None:
                    return
                await ws.send(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            raise
        except Exception:
            # A dead connection surfaces through messages() ending; there's
            # nothing useful to do from the keepalive path.
            pass

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("DeepgramLiveClient.connect() must be called before sending audio")
        await self._ws.send(chunk)

    async def finalize(self) -> None:
        """Asks Deepgram to flush whatever it's holding as a final result
        immediately, instead of waiting out the endpointing silence window.
        Used when the client stops talking so the last utterance isn't lost
        or delayed at the end of a session."""
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "Finalize"}))
        except Exception:
            pass

    async def messages(self) -> AsyncIterator[dict]:
        """Yields decoded JSON messages from Deepgram until the connection closes."""
        if self._ws is None:
            raise RuntimeError("DeepgramLiveClient.connect() must be called before reading messages")
        async for raw in self._ws:
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Dropped non-JSON message from Deepgram")
                continue

    async def close(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._ws is None:
            return
        ws, self._ws = self._ws, None
        try:
            await ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
