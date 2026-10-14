"""Ponytail-minimum tests for the pure arbitration/planning logic in
app.stt_session + app.translator. No network, no Deepgram.

Run from backend/:  .venv312/Scripts/python.exe -m unittest tests.test_stt_session -v
"""

import unittest

from app.stt_session import (
    Hypothesis,
    MultiLanguageSTT,
    TranscriptEvent,
    _overlaps,
    plan_connections,
)
from app.translator import pick_model


def hyp(language: str, text: str, conf: float = 0.9, start: float = 0.0, end: float = 1.0,
        words: list[dict] | None = None, detected: str | None = None) -> Hypothesis:
    return Hypothesis(
        language=language, detected_language=detected or language, text=text,
        confidence=conf, start=start, end=end, words=words or [],
    )


class PlanConnectionsTest(unittest.TestCase):
    def test_all_multi_collapse_to_one(self):
        plans, dropped = plan_connections(["en", "es", "fr"])
        self.assertEqual(len(plans), 1)
        self.assertTrue(plans[0].is_multi)
        self.assertEqual(plans[0].language, "multi")
        self.assertEqual(dropped, [])

    def test_outliers_get_own_connections(self):
        plans, dropped = plan_connections(["en", "es", "ar", "tl"])
        # en+es code-switch natively -> one multi connection; ar/tl are outliers.
        self.assertEqual([p.language for p in plans], ["multi", "ar", "tl"])
        self.assertEqual(dropped, [])

    def test_single_multi_language_stays_single(self):
        # Only ONE candidate is multi-code-switchable -> a plain single-language
        # connection (multi adds nothing with one language).
        plans, _ = plan_connections(["en", "ar", "tl"])
        self.assertEqual([p.language for p in plans], ["en", "ar", "tl"])

    def test_ceiling_drops_extras_but_keeps_multi(self):
        plans, dropped = plan_connections(["en", "es", "ar", "tl", "ta", "tr", "ko"])
        self.assertEqual(len(plans), 4)  # ceiling
        self.assertTrue(plans[0].is_multi)
        self.assertEqual(dropped, ["tr", "ko"])

    def test_unsupported_falls_back_to_en(self):
        plans, _ = plan_connections(["xx", "yy"])
        self.assertEqual([p.language for p in plans], ["en"])


class OverlapTest(unittest.TestCase):
    def test_same_utterance_matches(self):
        self.assertTrue(_overlaps(0.0, 2.0, 0.1, 1.9))

    def test_disjoint_does_not(self):
        self.assertFalse(_overlaps(0.0, 1.0, 1.5, 3.0))

    def test_barely_touching_does_not(self):
        # 0.1s intersection on a 2s shorter window = 5% < 50% threshold.
        self.assertFalse(_overlaps(0.0, 2.0, 1.9, 4.0))


class SpeakerRunsTest(unittest.TestCase):
    def make_stt(self, anchor: str = "en", with_timeline: bool = True) -> MultiLanguageSTT:
        plans, _ = plan_connections(["en"])
        stt = MultiLanguageSTT("key", "m", plans)
        stt._anchor_language = anchor
        if with_timeline:
            stt._speaker_timeline = [
                (0.0, 0.5, 0), (0.5, 1.0, 1), (1.0, 1.5, 0), (1.5, 2.0, 1),
            ]
        return stt

    def test_single_speaker_returns_none(self):
        stt = self.make_stt(with_timeline=False)
        stt._speaker_timeline = [(0.0, 2.0, 0)]
        w = hyp("fr", "bonjour", start=0.0, end=2.0, words=[
            {"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0},
        ])
        self.assertIsNone(stt._speaker_runs(w))

    def test_concurrent_speakers_split_into_runs(self):
        stt = self.make_stt()
        w = hyp("fr", "salut ca va oui et toi", start=0.0, end=2.0, words=[
            {"start": 0.0, "end": 0.5},   # -> speaker 0
            {"start": 0.5, "end": 1.0},   # -> speaker 1
            {"start": 1.0, "end": 1.5},   # -> speaker 0
            {"start": 1.5, "end": 2.0},   # -> speaker 1
        ])
        runs = stt._speaker_runs(w)
        self.assertEqual([s for s, _ in runs], [0, 1, 0, 1])
        self.assertEqual([len(ws) for _, ws in runs], [1, 1, 1, 1])

    def test_anchor_words_used_directly(self):
        stt = self.make_stt(anchor="en", with_timeline=False)
        w = hyp("en", "hello there hi", start=0.0, end=1.5, words=[
            {"start": 0.0, "end": 0.5, "speaker": 0},
            {"start": 0.5, "end": 1.0, "speaker": 0},
            {"start": 1.0, "end": 1.5, "speaker": 1},
        ])
        runs = stt._speaker_runs(w)
        self.assertEqual([s for s, _ in runs], [0, 1])
        self.assertEqual([len(ws) for _, ws in runs], [2, 1])

    def test_disabled_diarization_returns_none(self):
        stt = self.make_stt()
        stt._anchor_language = None
        w = hyp("fr", "salut", start=0.0, end=1.0, words=[{"start": 0.0, "end": 1.0}])
        self.assertIsNone(stt._speaker_runs(w))


class PickModelTest(unittest.TestCase):
    def test_auto_covers_all_babel_languages(self):
        for code in ("en", "es", "fr", "de", "zh", "ja", "ko", "pt", "hi", "ar", "tr", "tl", "ta"):
            self.assertIn("qwen", pick_model([code], "English"))

    def test_mode_quality_pins_quality(self):
        self.assertIn("gpt-oss-120b", pick_model(["en"], "English", "quality"))

    def test_unknown_target_falls_back_to_quality(self):
        # Target names come from the validated UI dropdown; an unknown one is
        # out-of-spec, so auto conservatively picks the quality model.
        self.assertIn("gpt-oss-120b", pick_model(["en"], "Klingon"))


class EarlyEmitTest(unittest.IsolatedAsyncioTestCase):
    async def test_window_resolves_when_all_connections_contributed(self):
        plans, _ = plan_connections(["en", "ar"])
        stt = MultiLanguageSTT("key", "m", plans, arbitration_grace_ms=10_000)
        # Fake both clients as up.
        stt._clients = {"multi": object(), "ar": object()}

        emitted: list[TranscriptEvent] = []

        async def fake_emit(window):
            emitted.append(TranscriptEvent(is_final=True, text="x", language="en"))
        stt._emit = fake_emit

        await stt._add_final(hyp("multi", "hello", detected="en"))
        self.assertEqual(len(emitted), 0)  # only one of two so far
        await stt._add_final(hyp("ar", "اهلا"))
        self.assertEqual(len(emitted), 1)  # resolved immediately, no 10s grace


if __name__ == "__main__":
    unittest.main()
