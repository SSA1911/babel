"""End-to-end test of /ws/stream: feed two SAPI voices (male then female)
through the real server, verify transcript + speaker labels + translation
deltas + timing fields.

Run from backend/:  .venv312/Scripts/python.exe scripts/e2e_stream_test.py
"""

import asyncio
import json
import os
import time
import wave

import websockets

SCRATCH = os.path.join(os.environ["LOCALAPPDATA"], "hermes", "cache", "scratch")
CHUNK = 3200  # 100ms of 16k mono PCM16


def load_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes())


async def main() -> None:
    t0 = time.monotonic()
    uri = "ws://localhost:8000/ws/stream?source_langs=en&target_language=Spanish"
    async with websockets.connect(uri, max_size=None) as ws:

        async def reader() -> None:
            while True:
                msg = json.loads(await ws.recv())
                mtype = msg.get("type")
                if mtype == "status":
                    print("STATUS:", json.dumps({k: v for k, v in msg.items() if k != "type"}))
                    if msg.get("data") == "language_switched":
                        continue
                    if msg.get("data") == "connected":
                        continue
                elif mtype == "transcript":
                    print(f"  t+{(time.monotonic()-t0)*1000:6.0f}ms TRANSCRIPT seg={msg['segment_id']} "
                          f"final={msg['is_final']} speaker={msg.get('speaker')} lang={msg.get('language')}: {msg['data']!r}")
                elif mtype == "translation":
                    print(f"  t+{(time.monotonic()-t0)*1000:6.0f}ms TRANSLATION seg={msg['segment_id']} "
                          f"final={msg['final']} ttft={msg.get('ms_to_first_token')} total={msg.get('ms_total')}: "
                          f"{msg['data'][:80]!r}")
                elif mtype == "error":
                    print("  ERROR:", msg.get("data"))

        r = asyncio.create_task(reader())
        for name in ("speakerA.wav", "speakerB.wav"):
            audio = load_pcm16(os.path.join(SCRATCH, name))
            for i in range(0, len(audio), CHUNK):
                await ws.send(audio[i:i + CHUNK])
                await asyncio.sleep(0.1)
            await asyncio.sleep(1.5)  # let endpointing + arbitration settle
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(4.0)  # drain flush + final translation
        r.cancel()


if __name__ == "__main__":
    asyncio.run(main())
