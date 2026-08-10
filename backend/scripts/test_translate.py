"""Manual smoke test for /ws/translate, run before wiring a client to it.

Usage:
    python scripts/test_translate.py "Hello, how are you?" Spanish
"""

import asyncio
import json
import sys

import websockets


async def main(text: str, target_lang: str, url: str = "ws://localhost:8000/ws/translate") -> None:
    async with websockets.connect(url) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            print(msg)
            if msg.get("type") == "status":
                break

        await ws.send(json.dumps({"segment_id": 1, "text": text, "target_lang": target_lang}))

        async for raw in ws:
            msg = json.loads(raw)
            print(msg)
            if msg.get("type") == "error":
                break
            if msg.get("type") == "translation" and msg.get("final"):
                break


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, how are you?"
    target_lang = sys.argv[2] if len(sys.argv) > 2 else "Spanish"
    asyncio.run(main(text, target_lang))
