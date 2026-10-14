"""One-off probe: TTFT + translation quality for candidate Groq models.

Tests each candidate as the Babel fast-lane translator: streaming, first-token
latency, and whether output is plausibly correct for a few representative
pairs (incl. languages llama-3.3 covered and ones it didn't).

Run: .venv312/Scripts/python.exe scripts/probe_models.py
"""

import asyncio
import os
import time

from dotenv import dotenv_values
from groq import AsyncGroq

KEY = dotenv_values(".env")["GROQ_API_KEY"]
client = AsyncGroq(api_key=KEY)

CANDIDATES = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]

CASES = [
    # (source_text, expected gist in output language) -- en->es, en->ar, ar->en
    ("I'll meet you at the site office tomorrow morning, don't be late.", "Spanish"),
    ("I'll meet you at the site office tomorrow morning, don't be late.", "Arabic"),
    ("صباح الخير، وين الوثيقة اللي طلبتها امس؟", "English"),
    ("Ang bagal ng trabaho ngayon, kulang pa yung materyales.", "English"),
]


async def probe(model: str, text: str, target: str) -> tuple[float, str]:
    t0 = time.monotonic()
    ttft = None
    out = []
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are a live interpreter. Translate the user's message into {target}. Output only the translation - no notes, quotes, or explanation. Keep the speaker's tone and register."},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        max_tokens=512,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning", None):
            continue
        if delta.content and ttft is None:
            ttft = time.monotonic() - t0
        if delta.content:
            out.append(delta.content)
    return ttft, "".join(out).strip()


async def main():
    for model in CANDIDATES:
        print(f"\n=== {model} ===")
        for text, target in CASES:
            try:
                ttft, out = await probe(model, text, target)
                print(f"  [{target:7s}] ttft={ttft*1000:5.0f}ms  {out[:110]}")
            except Exception as exc:
                msg = str(exc).replace("\n", " ")[:150]
                print(f"  [{target:7s}] FAILED: {msg}")


if __name__ == "__main__":
    asyncio.run(main())
