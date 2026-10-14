"""Probe: does Deepgram streaming accept diarize=true with language=multi,
and do results carry per-word speaker tags in multi mode?

Streams two SAPI-generated voices (male then female) into one connection in
each mode and prints every final's words: speaker + language + text.

Run from backend/:  .venv312/Scripts/python.exe scripts/probe_diarize_multi.py
"""

import asyncio
import json
import os
import sys
import time
import wave
from urllib.parse import urlencode

from dotenv import dotenv_values
from websockets.asyncio.client import connect

KEY = dotenv_values(".env")["DEEPGRAM_API_KEY"]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings  # noqa: E402  (STT model comes from the app's own config)

STT_MODEL = settings.deepgram_stt_model
SCRATCH = os.path.join(os.environ["LOCALAPPDATA"], "hermes", "cache", "scratch")
CHUNK = 3200  # 100ms of 16k mono PCM16


def load_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, (
            w.getframerate(), w.getnchannels(), w.getsampwidth())
        return w.readframes(w.getnframes())


async def probe(label: str, language: str, diarize: bool) -> None:
    params = {
        "encoding": "linear16", "sample_rate": "16000", "channels": "1",
        "interim_results": "true", "endpointing": "200", "utterance_end_ms": "1000",
        "smart_format": "true", "model": STT_MODEL, "language": language,
    }
    if diarize:
        params["diarize"] = "true"
    url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"

    print(f"\n=== {label}: language={language} diarize={diarize} ===")
    try:
        ws = await connect(url, additional_headers={"Authorization": f"Token {KEY}"}, max_size=None)
    except Exception as exc:
        print(f"  HANDSHAKE REJECTED: {exc}")
        return
    print("  handshake accepted")

    async def reader():
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") != "Results":
                    continue
                alt = (msg.get("channel") or {}).get("alternatives") or [{}]
                words = [(w.get("punctuated_word") or w.get("word"), w.get("speaker"), w.get("language"))
                         for w in alt[0].get("words") or []]
                print(f"  FINAL lang={alt[0].get('language', '-')!r} text={alt[0].get('transcript', '')!r}")
                print(f"    words: {words}")
        except Exception:
            pass

    r = asyncio.create_task(reader())
    for name in ("speakerA.wav", "speakerB.wav"):
        audio = load_pcm16(os.path.join(SCRATCH, name))
        for i in range(0, len(audio), CHUNK):
            await ws.send(audio[i:i + CHUNK])
            await asyncio.sleep(0.1)  # ~1x realtime pacing
        await asyncio.sleep(1.2)  # let endpointing finalize between speakers
    await ws.send(json.dumps({"type": "Finalize"}))
    await asyncio.sleep(1.0)
    await ws.close()
    try:
        await asyncio.wait_for(r, timeout=3)
    except Exception:
        pass


async def main():
    await probe("control (single-language)", "en", True)
    await probe("question under test", "multi", True)


if __name__ == "__main__":
    asyncio.run(main())
