"""Side-by-side quality probe: qwen3.8-27b vs gpt-oss-120b for Babel's
languages. Authored sentences with English reference glosses; each model
translates X->English and English->X; output is printed for manual scoring
plus TTFT. Throttled for free-tier rate limits.

Run from backend/:  .venv312/Scripts/python.exe scripts/probe_quality_compare.py
"""

import asyncio
import os
import time

from dotenv import dotenv_values
from groq import AsyncGroq

client = AsyncGroq(api_key=dotenv_values(".env")["GROQ_API_KEY"])

MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]

# (source_lang_name, sentence, reference English gloss)
CASES = [
    ("Spanish", "El Supervisor pidió que reprogramáramos la_offsets para el domingo.", None),  # typo trap
    ("Spanish", "No vamos a poder cumplir con el plazo, falta material en obra.", "We won't be able to meet the deadline, materials are missing on site."),
    ("Arabic", "خلينا نأجل الاجتماع لبكرة لأن المهندس م耽ه موجود.", None),  # code-switch-ish typo trap
    ("Arabic", "الشحنة وصلت متأخرة يومين، والخلط اتوقف على الموقع.", "The shipment arrived two days late, and the concrete mixing stopped on site."),
    ("Tagalog", "Kailangan nating tapusin ang rebars bago dumating ang inspektor.", "We need to finish the rebar before the inspector arrives."),
    ("Tamil", "நாளை காலையில் கான்கிரீட் ஊத்துவது நிற்கிறது, மழை வருகிறது.", "Concrete pouring is stopped tomorrow morning, rain is coming."),
    ("Chinese", "明天早上停工，混凝土车进不来。", "Work stops tomorrow morning, the concrete trucks can't get in."),
    ("Japanese", "足場の組み立ては来週の月曜日までかかります。", "Scaffold assembly will take until next Monday."),
    ("Korean", "철근 납품이 수요일로 밀렸어요.", "The rebar delivery has been pushed to Wednesday."),
    ("Turkish", "Şantiye şefi izinde, bugün kimseye imza attırmayın.", "The site chief is on leave, don't have anyone sign anything today."),
    ("Hindi", "सामग्री की कमी है, ठेकेदार से बात करनी पड़ेगी।", "There's a shortage of materials, we'll have to talk to the contractor."),
    ("Portuguese", "A betonagem fica suspensa até segunda-feira.", "The concrete pouring is suspended until Monday."),
    ("French", "Le coulage du béton est reporté à jeudi, préviens l'équipe.", "The concrete pour is postponed to Thursday, let the team know."),
    ("German", "Die Baustelle bleibt morgen wegen des Sturms geschlossen.", "The site stays closed tomorrow because of the storm."),
    # English -> X directions (target_language != English sessions)
    ("Spanish(en->es)", "The shipment arrived two days late, and the mixing stopped on site.", "es: La carga llegó dos días tarde, y el mezclado se detuvo en la obra."),
    ("Arabic(en->ar)", "We need more workers tomorrow for the pour.", "ar: نحتاج عمّال إضافيين غداً للصبّ."),
]


async def probe(model: str, text: str, target: str) -> tuple[float | None, str]:
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
    # (code, name) for the two English->X directions
    code_to_name = {"es": "Spanish", "ar": "Arabic"}
    for model in MODELS:
        print(f"\n===== {model} =====")
        for src, text, ref in CASES:
            if "(en->" in src:
                code = src.split("(en->")[1].rstrip(")")
                target = code_to_name[code]
            else:
                target = "English"
            try:
                ttft, out = await probe(model, text, target)
                ref_str = f"  [ref: {ref}]" if ref else "  [ref: none - typo/edge case, judge yourself]"
                print(f"[{src}] ttft={ttft*1000 if ttft else 0:4.0f}ms\n  OUT: {out}\n{ref_str}\n")
            except Exception as exc:
                print(f"[{src}] FAILED: {str(exc).replace(chr(10), ' ')[:140]}\n")
            await asyncio.sleep(2.0)  # free-tier friendly


if __name__ == "__main__":
    asyncio.run(main())
