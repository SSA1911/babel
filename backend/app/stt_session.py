"""Multi-connection STT session: several Deepgram connections listening to the
same audio at once, presented to the rest of the app as one event stream.

Deepgram's native code-switching (`language=multi`) resolves mixed-language
speech within a single utterance -- it even tags each word with its own
language -- but only across ten languages (see MULTI_MODE_LANGUAGES). Arabic,
Tagalog, Tamil, Turkish, Korean and Chinese sit outside it and must each be
transcribed by a single-language model.

The old approach was to run one connection and let the user re-point it with a
manual picker. That is what this module removes. Instead every language the
session might hear gets a connection, all of them receive the same audio, and
each finalized utterance is decided after the fact by comparing what the
connections heard. There is no chicken-and-egg problem -- the correct-language
model is always already listening, which is exactly why detecting a switch
from already-transcribed text never worked.

The cost is one Deepgram stream per connection while listening, so
plan_connections() collapses everything it can into a single `multi`
connection first and only fans out for genuine outliers.
"""

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from .deepgram_client import DeepgramLiveClient

logger = logging.getLogger("babel.stt")

# Nova-3 language codes (https://developers.deepgram.com/docs/models-languages-overview,
# checked 2026-08). Malayalam ("ml") is deliberately excluded -- confirmed
# absent from Deepgram's supported list. "tl" (Tagalog) is documented and works.
SUPPORTED_SOURCE_LANGUAGES = {
    "en", "es", "fr", "de", "zh", "ja", "ko", "pt", "hi", "ar", "tr", "tl", "ta",
}

# The set `language=multi` actually covers for Nova-3. Two or more of these in
# one session collapse into a single connection that code-switches natively,
# including mid-sentence, with per-word language tags.
MULTI_MODE_LANGUAGES = {"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"}

# Added to a hypothesis's score when it matches the language currently being
# spoken. Stops the winner flapping between two similarly-confident models on
# consecutive utterances.
STICKINESS_BONUS = 0.05
# Confidence is unreliable on very short utterances ("yes", "okay"), where
# both models often score high on nonsense. Lean harder on stickiness there.
SHORT_SEGMENT_WORDS = 2
SHORT_SEGMENT_STICKINESS_MULTIPLIER = 3.0

# Two finals belong to the same utterance if they overlap by more than this
# fraction of the shorter one.
WINDOW_OVERLAP_RATIO = 0.5


@dataclass(frozen=True)
class ConnectionPlan:
    """One Deepgram connection and the languages it is responsible for."""

    language: str          # what to send as `language=` ("multi" or a code)
    covers: tuple[str, ...]  # candidate languages this connection handles
    is_multi: bool


@dataclass
class Hypothesis:
    """One connection's version of a single utterance."""

    language: str          # the connection's language
    detected_language: str  # per-result language (differs from the above in multi mode)
    text: str
    confidence: float
    start: float
    end: float
    words: list[dict] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.words) or len(self.text.split())


@dataclass
class TranscriptEvent:
    """A resolved utterance, or a passthrough interim."""

    is_final: bool
    text: str
    language: str
    start: float = 0.0
    end: float = 0.0
    speaker: Optional[int] = None
    word_languages: Optional[list[dict]] = None
    # The best competing transcription of the same audio, kept so the
    # translator can reconstruct code-switched speech that no single
    # single-language model could transcribe correctly on its own.
    runner_up: Optional[Hypothesis] = None
    margin: float = 1.0


def plan_connections(
    candidates: list[str], baseline: int = 2, ceiling: int = 4
) -> tuple[list[ConnectionPlan], list[str]]:
    """Decides which Deepgram connections a session needs.

    Languages Deepgram can code-switch between collapse into one `multi`
    connection; everything else gets its own. `baseline` is documentation of
    the expected cost rather than a limit -- a session opens as many
    connections as its languages genuinely need. `ceiling` is a runaway guard.

    Returns (plans, dropped_languages). Anything dropped stays reachable
    through the manual picker.
    """
    seen: list[str] = []
    for code in candidates:
        if code in SUPPORTED_SOURCE_LANGUAGES and code not in seen:
            seen.append(code)
    if not seen:
        seen = ["en"]

    multi_group = [c for c in seen if c in MULTI_MODE_LANGUAGES]
    outliers = [c for c in seen if c not in MULTI_MODE_LANGUAGES]

    plans: list[ConnectionPlan] = []
    if len(multi_group) >= 2:
        plans.append(ConnectionPlan("multi", tuple(multi_group), True))
    elif multi_group:
        plans.append(ConnectionPlan(multi_group[0], (multi_group[0],), False))
    for code in outliers:
        plans.append(ConnectionPlan(code, (code,), False))

    dropped: list[str] = []
    if len(plans) > ceiling:
        for extra in plans[ceiling:]:
            dropped.extend(extra.covers)
        plans = plans[:ceiling]
        logger.warning(
            "Session requested %d STT connections, ceiling is %d; %s will need the manual picker",
            len(plans) + len(dropped), ceiling, ", ".join(dropped),
        )
    elif len(plans) > baseline:
        logger.info(
            "Session is opening %d STT connections (baseline %d) for languages %s",
            len(plans), baseline, ", ".join(seen),
        )
    return plans, dropped


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    intersection = min(a_end, b_end) - max(a_start, b_start)
    if intersection <= 0:
        return False
    shorter = min(a_end - a_start, b_end - b_start)
    if shorter <= 0:
        return True
    return (intersection / shorter) >= WINDOW_OVERLAP_RATIO


@dataclass
class _Window:
    """Competing transcriptions of one utterance, awaiting arbitration."""

    start: float
    end: float
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    timer: Optional[asyncio.Task] = None


class MultiLanguageSTT:
    """Owns every Deepgram connection for one client session.

    push_audio() broadcasts to all of them; events() yields one merged,
    arbitrated stream of TranscriptEvents.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        plans: list[ConnectionPlan],
        endpointing_ms: int = 100,
        utterance_end_ms: int = 1000,
        arbitration_grace_ms: int = 250,
        diarize: bool = True,
    ):
        self._api_key = api_key
        self._model = model
        self._plans = plans
        self._endpointing_ms = endpointing_ms
        self._utterance_end_ms = utterance_end_ms
        self._grace = arbitration_grace_ms / 1000.0
        self._want_diarize = diarize

        self._clients: dict[str, DeepgramLiveClient] = {}
        self._readers: list[asyncio.Task] = []
        self._events: "asyncio.Queue[TranscriptEvent | None]" = asyncio.Queue()
        self._windows: list[_Window] = []
        self._lock = asyncio.Lock()

        # Word-level (start, end, speaker) from the anchor connection. Every
        # segment's speaker label is read from here regardless of which
        # connection won, so labels stay consistent across languages.
        self._speaker_timeline: list[tuple[float, float, int]] = []

        self._anchor_language: Optional[str] = None
        self._current_language: str = plans[0].covers[0] if plans else "en"
        self._closed = False

    # -- lifecycle -------------------------------------------------------

    @property
    def connection_count(self) -> int:
        return len(self._clients)

    @property
    def current_language(self) -> str:
        return self._current_language

    @property
    def auto_languages(self) -> list[str]:
        return [code for plan in self._plans for code in plan.covers]

    @property
    def diarization_enabled(self) -> bool:
        return self._want_diarize and self._anchor_language is not None

    async def swap_in_language(self, language: str) -> None:
        """Replaces the last connection with one for `language`.

        Only reachable when a session selected more languages than
        STT_CONNECTION_CEILING allows connections for -- everything within the
        ceiling is detected automatically and never needs switching. The
        anchor connection is never swapped, so speaker labels stay continuous.
        """
        if language not in SUPPORTED_SOURCE_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        if any(language in plan.covers for plan in self._plans):
            return

        victim = next(
            (p for p in reversed(self._plans) if p.language != self._anchor_language), None
        )
        if victim is None:
            raise RuntimeError("No swappable connection available")

        replacement = DeepgramLiveClient(
            self._api_key,
            model=self._model,
            language=language,
            diarize=False,
            endpointing_ms=self._endpointing_ms,
            utterance_end_ms=self._utterance_end_ms,
        )
        await replacement.connect()

        old = self._clients.pop(victim.language, None)
        self._plans = [p for p in self._plans if p is not victim]
        self._plans.append(ConnectionPlan(language, (language,), False))
        self._clients[language] = replacement

        for task in list(self._readers):
            if task.get_name() == f"dg_reader_{victim.language}":
                task.cancel()
                self._readers.remove(task)
        if old is not None:
            await old.close()

        self._current_language = language
        self._readers.append(
            asyncio.create_task(self._read(language, replacement), name=f"dg_reader_{language}")
        )
        logger.info("Swapped STT connection %s -> %s", victim.language, language)

    def _choose_anchor(self) -> ConnectionPlan:
        """Diarization runs on exactly one connection. Prefer a
        single-language one: `diarize` combined with `language=multi` is not a
        documented combination, so avoid relying on it when there's a choice."""
        for plan in self._plans:
            if not plan.is_multi:
                return plan
        return self._plans[0]

    async def start(self) -> None:
        anchor = self._choose_anchor()
        self._anchor_language = anchor.language

        async def build(plan: ConnectionPlan) -> tuple[str, DeepgramLiveClient]:
            wants_diarize = self._want_diarize and plan.language == anchor.language
            client = DeepgramLiveClient(
                self._api_key,
                model=self._model,
                language=plan.language,
                diarize=wants_diarize,
                endpointing_ms=self._endpointing_ms,
                utterance_end_ms=self._utterance_end_ms,
            )
            try:
                await client.connect()
            except Exception:
                if not wants_diarize:
                    raise
                # Deepgram doesn't document diarize+multi together. If the
                # anchor had to be a multi connection and it was rejected,
                # fall back to transcription without speaker labels rather
                # than failing the whole session.
                logger.warning(
                    "Deepgram rejected diarize on language=%s; retrying without "
                    "speaker labels for this session",
                    plan.language,
                )
                client = DeepgramLiveClient(
                    self._api_key,
                    model=self._model,
                    language=plan.language,
                    diarize=False,
                    endpointing_ms=self._endpointing_ms,
                    utterance_end_ms=self._utterance_end_ms,
                )
                await client.connect()
                self._anchor_language = None
            return plan.language, client

        # Connect in parallel so every connection's internal clock starts at
        # roughly the same moment -- window overlap matching depends on their
        # timelines agreeing.
        results = await asyncio.gather(*(build(p) for p in self._plans))
        self._clients = dict(results)

        for language, client in self._clients.items():
            self._readers.append(
                asyncio.create_task(self._read(language, client), name=f"dg_reader_{language}")
            )
        logger.info(
            "STT session up: %d connection(s) [%s], diarization anchor=%s",
            len(self._clients), ", ".join(self._clients), self._anchor_language or "none",
        )

    async def push_audio(self, chunk: bytes) -> None:
        for language, client in list(self._clients.items()):
            try:
                await client.send_audio(chunk)
            except Exception:
                # One dead connection shouldn't stop the others from hearing
                # the rest of the conversation.
                logger.debug("Dropped audio chunk for %s", language)

    async def finalize(self) -> None:
        await asyncio.gather(
            *(c.finalize() for c in self._clients.values()), return_exceptions=True
        )

    async def close(self) -> None:
        self._closed = True
        for task in self._readers:
            task.cancel()
        async with self._lock:
            for window in self._windows:
                if window.timer is not None:
                    window.timer.cancel()
            self._windows.clear()
        await asyncio.gather(*self._readers, return_exceptions=True)
        await asyncio.gather(
            *(c.close() for c in self._clients.values()), return_exceptions=True
        )
        await self._events.put(None)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                break
            yield event

    # -- reading ---------------------------------------------------------

    async def _read(self, language: str, client: DeepgramLiveClient) -> None:
        try:
            async for message in client.messages():
                if message.get("type") not in (None, "Results"):
                    continue
                channel = message.get("channel") or {}
                alternatives = channel.get("alternatives") or []
                if not alternatives:
                    continue
                alt = alternatives[0]
                text = (alt.get("transcript") or "").strip()
                if not text:
                    continue

                words = alt.get("words") or []
                start = float(message.get("start", 0.0))
                duration = float(message.get("duration", 0.0))
                detected = self._detected_language(language, alt, words)

                if not message.get("is_final", False):
                    # Only the active language's interims reach the UI --
                    # otherwise the preview line flickers between two
                    # competing transcriptions of the same half-spoken word.
                    if self._is_active_connection(language):
                        await self._events.put(
                            TranscriptEvent(
                                is_final=False, text=text, language=detected,
                                start=start, end=start + duration,
                            )
                        )
                    continue

                # Finals only: interims restate the same words repeatedly, and
                # recording those would count one word many times in the
                # speaker-overlap vote (and grow the timeline without bound).
                if language == self._anchor_language:
                    self._record_speakers(words)

                await self._add_final(
                    Hypothesis(
                        language=language,
                        detected_language=detected,
                        text=text,
                        confidence=float(alt.get("confidence", 0.0) or 0.0),
                        start=start,
                        end=start + duration,
                        words=words,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closed:
                logger.exception("Deepgram reader for %s failed", language)

    def _is_active_connection(self, language: str) -> bool:
        if len(self._clients) == 1:
            return True
        plan = next((p for p in self._plans if p.language == language), None)
        return bool(plan and self._current_language in plan.covers)

    @staticmethod
    def _detected_language(language: str, alt: dict, words: list[dict]) -> str:
        """In multi mode Deepgram reports the language it actually heard;
        otherwise it's whatever the connection was opened with."""
        if language != "multi":
            return language
        languages = alt.get("languages")
        if languages:
            return languages[0]
        for word in words:
            if word.get("language"):
                return word["language"]
        return language

    def _record_speakers(self, words: list[dict]) -> None:
        for word in words:
            speaker = word.get("speaker")
            if speaker is None:
                continue
            try:
                self._speaker_timeline.append(
                    (float(word["start"]), float(word["end"]), int(speaker))
                )
            except (KeyError, TypeError, ValueError):
                continue
        # Bounded so a long session doesn't grow this without limit; an
        # utterance is only ever matched against recent history.
        if len(self._speaker_timeline) > 4000:
            del self._speaker_timeline[:2000]

    # -- arbitration -----------------------------------------------------

    async def _add_final(self, hypothesis: Hypothesis) -> None:
        # Nothing to arbitrate in a single-connection session, so don't pay
        # the grace delay for it.
        if len(self._clients) == 1:
            window = _Window(start=hypothesis.start, end=hypothesis.end)
            window.hypotheses[hypothesis.language] = hypothesis
            await self._emit(window)
            return

        async with self._lock:
            window = next(
                (
                    w for w in self._windows
                    if _overlaps(w.start, w.end, hypothesis.start, hypothesis.end)
                ),
                None,
            )
            if window is None:
                window = _Window(start=hypothesis.start, end=hypothesis.end)
                self._windows.append(window)
                window.timer = asyncio.create_task(self._resolve_after_grace(window))
            else:
                window.start = min(window.start, hypothesis.start)
                window.end = max(window.end, hypothesis.end)

            existing = window.hypotheses.get(hypothesis.language)
            # A connection can emit several finals inside one window; keep the
            # longest, which is the most complete version of the utterance.
            if existing is None or len(hypothesis.text) > len(existing.text):
                window.hypotheses[hypothesis.language] = hypothesis

    async def _resolve_after_grace(self, window: _Window) -> None:
        try:
            await asyncio.sleep(self._grace)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if window not in self._windows:
                return
            self._windows.remove(window)
        await self._emit(window)

    def _score(self, hypothesis: Hypothesis) -> float:
        confidences = [
            float(w["confidence"]) for w in hypothesis.words if w.get("confidence") is not None
        ]
        base = statistics.fmean(confidences) if confidences else hypothesis.confidence

        bonus = 0.0
        plan = next((p for p in self._plans if p.language == hypothesis.language), None)
        if plan and self._current_language in plan.covers:
            bonus = STICKINESS_BONUS
            if hypothesis.word_count <= SHORT_SEGMENT_WORDS:
                bonus *= SHORT_SEGMENT_STICKINESS_MULTIPLIER
        return base + bonus

    async def _emit(self, window: _Window) -> None:
        if not window.hypotheses:
            return
        ranked = sorted(window.hypotheses.values(), key=self._score, reverse=True)
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        margin = (self._score(winner) - self._score(runner_up)) if runner_up else 1.0

        self._current_language = winner.detected_language

        await self._events.put(
            TranscriptEvent(
                is_final=True,
                text=winner.text,
                language=winner.detected_language,
                start=winner.start,
                end=winner.end,
                speaker=self._speaker_for(winner),
                word_languages=self._word_languages(winner),
                runner_up=runner_up,
                margin=margin,
            )
        )

    def _speaker_for(self, hypothesis: Hypothesis) -> Optional[int]:
        """The speaker who holds the most of this utterance's duration.

        Labels always come from the anchor connection's timeline (or the
        winner's own words when the winner *is* the anchor), because speaker
        indices are per-connection and wouldn't agree across languages.
        Deepgram's diarization is acoustic, so the anchor's segmentation is
        equally valid for an utterance any connection transcribed."""
        if hypothesis.language == self._anchor_language:
            own = [w for w in hypothesis.words if w.get("speaker") is not None]
            if own:
                totals: dict[int, float] = {}
                for word in own:
                    try:
                        span = float(word["end"]) - float(word["start"])
                        totals[int(word["speaker"])] = totals.get(int(word["speaker"]), 0.0) + span
                    except (KeyError, TypeError, ValueError):
                        continue
                if totals:
                    return max(totals, key=totals.get)

        overlap: dict[int, float] = {}
        for start, end, speaker in self._speaker_timeline:
            intersection = min(end, hypothesis.end) - max(start, hypothesis.start)
            if intersection > 0:
                overlap[speaker] = overlap.get(speaker, 0.0) + intersection
        if not overlap:
            return None
        return max(overlap, key=overlap.get)

    @staticmethod
    def _word_languages(hypothesis: Hypothesis) -> Optional[list[dict]]:
        """Per-word language tags, present only on multi connections. This is
        what lets a single mixed-language sentence render with its real
        breakdown instead of one label for the whole utterance."""
        if hypothesis.language != "multi":
            return None
        tagged = [
            {"word": w.get("punctuated_word") or w.get("word", ""), "language": w["language"]}
            for w in hypothesis.words
            if w.get("language")
        ]
        if not tagged or len({t["language"] for t in tagged}) < 2:
            return None
        return tagged
