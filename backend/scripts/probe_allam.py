"""Probe: ALLaM-2-7B vs qwen3.8-27b on the Arabic translation lane.

ALLaM is bilingual (Arabic <-> English) only, so it can only ever serve
sessions where Arabic is in play and the target is English or Arabic.
Cases: ar->en (typical Babel use), en->ar, plus one dialect-heavy and one
code-switch-ish case. Scores TTFT + output for manual comparison.

Run from backend/:  .venv312/Scripts/python.exe scripts/probe_allam.py
"""

import asyncio
import time

from dotenv import dotenv_values
from groq import AsyncGroq

client = AsyncGroq(api_key=dotenv_values(".env")["GROQ_API_KEY"])

MODELS = ["allam-2-7b", "qwen/qwen3.8-27b"]

# Khaleeji/Gulf and Egyptian dialect cases (the user's actual Arabic), plus
# the two code-switch cases. Glosses are mine, for scoring.
CASES = [
    ("Khaleeji casual", "شخبارك؟ وين التقرير اللي قلت لي عنه امس؟ خلاص خلينا نبدأ الشغل.", "English",
     "How are you? Where's the report you told me about yesterday? Fine, let's start the work."),
    ("Khaleeji site", "ما نبغى نعيّد الصب، الخرسانة لسى ما جفت. خلها لبكرة الصبح.", "English",
     "We don't want to re-pour, the concrete hasn't dried yet. Leave it for tomorrow morning."),
    ("Khaleeji command", "خلّ العمال يخلصون التسليح قبل المغرب وبعدها نعطيكم الرست.", "English",
     "Have the workers finish the rebar before sunset, then we'll give you the rest."),
    ("Egyptian casual", "إزيك؟ الشغلانة دي هتاخد وقت، محتاجين عمال كمان ورصيد أسمنت.", "English",
     "How are you? This job will take time, we need more workers and a cement reserve."),
    ("Egyptian site", "المixer واصل؟ لأ؟ يبقى الصب النهارده مش هيتم، هنعوّض بكرة.", "English",
     "Has the mixer arrived? No? Then the pour won't happen today, we'll postpone to tomorrow."),
    ("Egyptian formal-ish", "من فضلك ابعت الـ invoice بتاعة الشهر ده قبل يوم الخميس.", "English",
     "Please send this month's invoice before Thursday."),
    ("ar->en code-switch", "الملف اللي في الـ folder الثاني محتاج revision قبل الـ deadline.", "English",
     "The file in the second folder needs revision before the deadline."),
    ("en->ar Gulf flavor", "We need more workers tomorrow for the concrete pour.", "Arabic",
     "نحتاج/نبي عمال إضافيين غداً لصب الخرسانة."),
]


async def probe(model: str, text: str, target: str) -> tuple[float | None, str, str | None]:
    t0 = time.monotonic()
    ttft = None
    out = []
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are a live interpreter. Translate the user's message into {target}. Output only the translation - no notes, quotes, or explanation. Keep the speaker's tone and register."},
            {"role": "user", "content": text},
        ],
        temperature=0.2, max_tokens=512, stream=True,
    )
    finish = None
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
        if chunk.choices and chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    return ttft, "".join(out).strip(), finish


async def main():
    for model in MODELS:
        print(f"\n===== {model} =====")
        for label, text, target, *ref in CASES:
            try:
                ttft, out, finish = await probe(model, text, target)
                ref_str = f"  [ref: {ref[0]}]" if ref else ""
                print(f"[{label}] ttft={ttft*1000 if ttft else 0:5.0f}ms finish={finish}\n  OUT: {out}\n{ref_str}\n")
            except Exception as exc:
                print(f"[{label}] FAILED: {str(exc).replace(chr(10), ' ')[:180]}\n")
            await asyncio.sleep(1.5)


if __name__ == "__main__":
    asyncio.run(main())
