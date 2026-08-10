"""Thin async wrapper around Deepgram's live transcription WebSocket API."""

import json
import logging
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

logger = logging.getLogger("babel.deepgram")

DEEPGRAM_LIVE_BASE_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramLiveClient:
    """Owns one persistent Deepgram streaming connection for a single client session."""

    def __init__(self, api_key: str, model: str = "nova-2", language: str = "en"):
        self._api_key = api_key
        self._model = model
        self._language = language
        self._ws: Optional[ClientConnection] = None

    async def connect(self) -> None:
        params = {
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
            "interim_results": "true",
            "endpointing": "300",
            "smart_format": "true",
            "model": self._model,
            "language": self._language,
        }
        url = f"{DEEPGRAM_LIVE_BASE_URL}?{urlencode(params)}"
        self._ws = await connect(
            url,
            additional_headers={"Authorization": f"Token {self._api_key}"},
            ping_interval=5,
            ping_timeout=20,
            max_size=None,
        )

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("DeepgramLiveClient.connect() must be called before sending audio")
        await self._ws.send(chunk)

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
