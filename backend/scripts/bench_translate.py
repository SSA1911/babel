"""Measures translation latency per Groq model, so model choice is settled by
numbers rather than by feel.

The metric that matters for live captioning is time-to-first-token: how long a
bubble sits on "translating…" before any text appears. Total time matters much
less, since the rest streams in as it arrives.

This calls Groq directly (no backend server needed). Run from backend/:

    python scripts/bench_translate.py
    python scripts/bench_translate.py --target Arabic --runs 5
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.translator import stream_translation  # noqa: E402

# Deliberately a mix: a short utterance where per-call overhead dominates, a
# normal conversational sentence, and a longer one.
SAMPLES = [
    "Where is the meeting room?",
    "I wanted to ask whether the delivery is still arriving on Thursday morning.",
    "We've reviewed the proposal and the team thinks the timeline is too tight, "
    "so we'd like to suggest moving the launch back by two weeks.",
]

MODELS = [
    ("gpt-oss-120b (quality)", "openai/gpt-oss-120b"),
    ("llama-3.3-70b (fast)", "llama-3.3-70b-versatile"),
    ("llama-3.1-8b", "llama-3.1-8b-instant"),
]

# What the app did before this was tuned: gpt-oss at its default reasoning
# effort, the longer system prompt, and -- the part that actually hurt --
# reading only delta.content, so the reasoning phase looked like dead air with
# no tokens at all. Measured here so the improvement is a number, not a claim.
BASELINE_PROMPT = (
    "You are a professional real-time interpreter embedded in a live captioning "
    "system. Translate the user's message from its detected source language into "
    "{target_language}. Output ONLY the direct translation. Do not include "
    "explanations, notes, pleasantries, disclaimers, or surrounding quotes. "
    "If the message is already in {target_language}, output it unchanged. "
    "Preserve the speaker's tone, register, and intent, and keep the translation "
    "as a single fluent utterance."
)


async def measure(text: str, target: str, model: str) -> tuple[float, float, str]:
    start = time.monotonic()
    first: float | None = None
    out = []
    async for delta in stream_translation(text, target, model=model):
        if first is None:
            first = time.monotonic()
        out.append(delta)
    end = time.monotonic()
    ttft = ((first or end) - start) * 1000
    return ttft, (end - start) * 1000, "".join(out)


async def measure_baseline(text: str, target: str) -> tuple[float, float, str]:
    """Reproduces the pre-refactor call exactly, including only counting
    `delta.content` as visible output."""
    from app.translator import _client

    start = time.monotonic()
    stream = await _client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": BASELINE_PROMPT.format(target_language=target)},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        stream=True,
    )
    first: float | None = None
    out = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            if first is None:
                first = time.monotonic()
            out.append(content)
    end = time.monotonic()
    return ((first or end) - start) * 1000, (end - start) * 1000, "".join(out)


async def main(target: str, runs: int, show_output: bool) -> None:
    print(f"target={target}  runs={runs}  (translation_max_tokens={settings.translation_max_tokens})\n")
    header = f"{'model':<32}{'first token':>14}{'total':>12}"
    print(header)
    print("-" * len(header))

    entries = [("BEFORE: gpt-oss default effort", None)] + MODELS
    for label, model in entries:
        ttfts, totals, sample_out = [], [], ""
        try:
            for _ in range(runs):
                for text in SAMPLES:
                    if model is None:
                        ttft, total, out = await measure_baseline(text, target)
                    else:
                        ttft, total, out = await measure(text, target, model)
                    ttfts.append(ttft)
                    totals.append(total)
                    sample_out = out
        except Exception as exc:
            print(f"{label:<32}{'FAILED':>14}   {type(exc).__name__}: {exc}")
            continue

        print(
            f"{label:<32}"
            f"{statistics.median(ttfts):>11.0f}ms"
            f"{statistics.median(totals):>11.0f}ms"
        )
        if show_output:
            print(f"{'':<32}last output: {sample_out[:70]}")

    print(
        "\nMedian across "
        f"{runs * len(SAMPLES)} calls per model. First-token time is what the user "
        "actually perceives as lag."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="English", help="target language name")
    parser.add_argument("--runs", type=int, default=3, help="passes over the sample set")
    parser.add_argument("--show-output", action="store_true", help="print a sample translation")
    args = parser.parse_args()
    asyncio.run(main(args.target, args.runs, args.show_output))
