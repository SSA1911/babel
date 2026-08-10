"""Manual smoke test for /ws/diarize, run before wiring a client to it.

Streams a 16kHz mono 16-bit PCM WAV file in real-time-paced chunks and
prints speaker_turn events as diart resolves them. A two-speaker recording
(e.g. two people alternating) is the most useful sanity check.

Usage:
    python scripts/test_diarize.py path/to/16k_mono.wav

If your source file isn't already 16kHz mono 16-bit PCM, convert it first:
    ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 16k_mono.wav
"""

import asyncio
import json
import sys
import wave

import websockets

CHUNK_MS = 50


async def main(wav_path: str, url: str = "ws://localhost:8000/ws/diarize") -> None:
    with wave.open(wav_path, "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise SystemExit(
                "Expected 16kHz mono 16-bit PCM WAV. Convert with: "
                "ffmpeg -i input.ext -ar 16000 -ac 1 -sample_fmt s16 16k_mono.wav"
            )
        chunk_frames = int(16000 * CHUNK_MS / 1000)

        async with websockets.connect(url) as ws:

            async def receiver() -> None:
                async for raw in ws:
                    print(json.loads(raw))

            recv_task = asyncio.create_task(receiver())

            frames = wf.readframes(chunk_frames)
            while frames:
                await ws.send(frames)
                await asyncio.sleep(CHUNK_MS / 1000)
                frames = wf.readframes(chunk_frames)

            print("...done sending audio, waiting for the final analysis window to resolve...")
            await asyncio.sleep(6)
            recv_task.cancel()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/test_diarize.py path/to/16k_mono.wav")
    asyncio.run(main(sys.argv[1]))
