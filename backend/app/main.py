"""FastAPI orchestrator: client audio <-> Deepgram STT <-> Groq translation."""

import asyncio
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import settings
from .schemas import TranslateRequest
from .stt_session import (
    SUPPORTED_SOURCE_LANGUAGES,
    MultiLanguageSTT,
    TranscriptEvent,
    plan_connections,
)
from .translator import pick_model, stream_translation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("babel.main")

app = FastAPI(title="Babel Live Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}


def _parse_candidate_languages(client_ws: WebSocket) -> list[str]:
    """Reads ?source_langs=en,ar,tl (multi-candidate) or falls back to the
    older single ?source_lang=en for backward compatibility."""
    multi = client_ws.query_params.get("source_langs")
    if multi:
        candidates = [
            c.strip() for c in multi.split(",") if c.strip() in SUPPORTED_SOURCE_LANGUAGES
        ]
        if candidates:
            return candidates

    single = client_ws.query_params.get("source_lang", settings.source_language)
    return [single if single in SUPPORTED_SOURCE_LANGUAGES else settings.source_language]


@dataclass
class _Segment:
    """One utterance in flight through translation.

    Translations run concurrently, but each writes into its own queue and a
    single emitter drains them in segment order -- so a slow response for
    segment 1 can no longer delay segment 2 *starting*, while the guarantee
    that translations render in the order they were spoken is preserved.
    """

    segment_id: int
    deltas: "asyncio.Queue[str | None]" = field(default_factory=asyncio.Queue)
    fused: bool = False
    t_final: float = 0.0
    t_first_token: Optional[float] = None


@app.websocket("/ws/stream")
async def stream_endpoint(client_ws: WebSocket):
    await client_ws.accept()

    target_language = client_ws.query_params.get("lang", settings.target_language)
    candidate_langs = _parse_candidate_languages(client_ws)
    model_mode = client_ws.query_params.get("model_mode")
    fusion_enabled = client_ws.query_params.get("fusion", "1") != "0"

    model = pick_model(candidate_langs, target_language, model_mode)

    plans, dropped = plan_connections(
        candidate_langs,
        baseline=settings.stt_connection_baseline,
        ceiling=settings.stt_connection_ceiling,
    )

    stt = MultiLanguageSTT(
        settings.deepgram_api_key,
        model=settings.deepgram_stt_model,
        plans=plans,
        endpointing_ms=settings.endpointing_ms,
        utterance_end_ms=settings.utterance_end_ms,
        arbitration_grace_ms=settings.arbitration_grace_ms,
    )
    try:
        await stt.start()
    except Exception as exc:
        logger.exception("Failed to start STT session")
        await _safe_send_json(
            client_ws, {"type": "error", "data": f"Could not connect to speech engine: {exc}"}
        )
        await _safe_close(client_ws)
        return

    await _safe_send_json(
        client_ws,
        {
            "type": "status",
            "data": "connected",
            "languages": stt.auto_languages,
            "manual_languages": dropped,
            "connections": stt.connection_count,
            "diarization": stt.diarization_enabled,
            "model": model,
        },
    )

    segment_ids = itertools.count(1)
    order_queue: "asyncio.Queue[_Segment | None]" = asyncio.Queue()
    translation_slots = asyncio.Semaphore(settings.translation_concurrency)
    session_ended = asyncio.Event()

    async def translate_segment(segment: _Segment, text: str, alternate: Optional[str]) -> None:
        try:
            async with translation_slots:
                async for delta in stream_translation(
                    text, target_language, model=model, alternate_text=alternate
                ):
                    if segment.t_first_token is None:
                        segment.t_first_token = time.monotonic()
                    await segment.deltas.put(delta)
        except Exception as exc:
            logger.exception("Translation failed for segment %s", segment.segment_id)
            await _safe_send_json(
                client_ws, {"type": "error", "data": f"Translation failed: {exc}"}
            )
        finally:
            await segment.deltas.put(None)

    async def emit_translations() -> None:
        """Drains segment queues strictly in spoken order."""
        while True:
            segment = await order_queue.get()
            if segment is None:
                break
            while True:
                delta = await segment.deltas.get()
                if delta is None:
                    break
                await _safe_send_json(
                    client_ws,
                    {
                        "type": "translation",
                        "segment_id": segment.segment_id,
                        "data": delta,
                        "final": False,
                    },
                )
            now = time.monotonic()
            ttft = (
                int((segment.t_first_token - segment.t_final) * 1000)
                if segment.t_first_token
                else None
            )
            total = int((now - segment.t_final) * 1000)
            logger.info(
                "segment %s translated in %sms (first token %sms)%s",
                segment.segment_id, total, ttft, " [fused]" if segment.fused else "",
            )
            await _safe_send_json(
                client_ws,
                {
                    "type": "translation",
                    "segment_id": segment.segment_id,
                    "data": "",
                    "final": True,
                    "ms_to_first_token": ttft,
                    "ms_total": total,
                    "fused": segment.fused,
                },
            )

    async def consume_transcripts() -> None:
        try:
            async for event in stt.events():
                if not event.is_final:
                    await _safe_send_json(
                        client_ws,
                        {
                            "type": "transcript",
                            "segment_id": -1,
                            "data": event.text,
                            "is_final": False,
                            "language": event.language,
                        },
                    )
                    continue

                seg_id = next(segment_ids)
                await _safe_send_json(
                    client_ws,
                    {
                        "type": "transcript",
                        "segment_id": seg_id,
                        "data": event.text,
                        "is_final": True,
                        "language": event.language,
                        "speaker": event.speaker,
                        "word_languages": event.word_languages,
                    },
                )

                alternate = _fusion_candidate(event, fusion_enabled)
                segment = _Segment(
                    segment_id=seg_id, fused=alternate is not None, t_final=time.monotonic()
                )
                await order_queue.put(segment)
                asyncio.create_task(
                    translate_segment(segment, event.text, alternate),
                    name=f"translate_{seg_id}",
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error consuming transcripts")
            session_ended.set()

    async def client_reader() -> None:
        """Binary frames are audio, forwarded to every Deepgram connection;
        text frames are JSON control messages."""
        try:
            while True:
                message = await client_ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    await stt.push_audio(data)
                    continue
                text = message.get("text")
                if text is not None:
                    await _handle_control_message(text, stt, dropped, client_ws)
        except WebSocketDisconnect:
            logger.info("Client disconnected (audio stream)")
        except Exception:
            logger.exception("Error in client reader")
        finally:
            # Flush whatever Deepgram is still holding so the last utterance
            # isn't lost waiting out the endpointing window.
            await stt.finalize()
            session_ended.set()

    tasks = [
        asyncio.create_task(consume_transcripts(), name="consume_transcripts"),
        asyncio.create_task(client_reader(), name="client_reader"),
        asyncio.create_task(emit_translations(), name="emit_translations"),
    ]

    try:
        await session_ended.wait()
    finally:
        await stt.close()
        # The client is gone by this point, so in-flight translations have
        # nowhere to render; drop them rather than waiting them out.
        await order_queue.put(None)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _safe_close(client_ws)
        logger.info("Session cleaned up")


def _fusion_candidate(event: TranscriptEvent, enabled: bool) -> Optional[str]:
    """Returns the runner-up transcript when the two recognizers disagreed
    closely enough that the speaker was probably code-switching.

    A clear winner means monolingual speech and the runner-up is noise; a
    near-tie means each recognizer got its own language right and mangled the
    other, and only both together determine what was said.
    """
    if not enabled or event.runner_up is None:
        return None
    if settings.fusion_margin <= 0:
        return None
    if event.margin > settings.fusion_margin:
        return None
    return event.runner_up.text


async def _handle_control_message(
    raw: str, stt: MultiLanguageSTT, dropped: list[str], client_ws: WebSocket
) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    if payload.get("type") != "switch_language":
        return
    language = payload.get("language", "")
    # Every language the session actually connected for is detected
    # automatically, so a switch request for one of those is a no-op. Only
    # languages pushed past the connection ceiling need this path.
    if language not in dropped:
        return
    try:
        await stt.swap_in_language(language)
    except Exception as exc:
        logger.exception("Failed to swap in %s", language)
        await _safe_send_json(
            client_ws, {"type": "error", "data": f"Could not switch to {language}: {exc}"}
        )
        return
    dropped[:] = [c for c in dropped if c != language]
    await _safe_send_json(
        client_ws,
        {"type": "status", "data": "language_switched", "language": language},
    )


@app.websocket("/ws/translate")
async def translate_endpoint(client_ws: WebSocket):
    """Accepts {"segment_id", "text", "target_lang"} JSON requests and streams
    back translation deltas, reusing stream_translation(). Decoupled from STT
    so a client that already has transcribed text can get a cloud translation
    without re-uploading audio. A single worker processes requests in order,
    so responses can't interleave."""
    await client_ws.accept()
    await _safe_send_json(client_ws, {"type": "status", "data": "connected"})

    queue: "asyncio.Queue[TranslateRequest | None]" = asyncio.Queue()

    async def receive_requests() -> None:
        try:
            while True:
                raw = await client_ws.receive_json()
                try:
                    req = TranslateRequest.model_validate(raw)
                except ValidationError as exc:
                    await _safe_send_json(
                        client_ws, {"type": "error", "data": f"Invalid request: {exc}"}
                    )
                    continue
                await queue.put(req)
        except WebSocketDisconnect:
            logger.info("Client disconnected (translate stream)")
        except Exception:
            logger.exception("Error reading translate requests")
        finally:
            await queue.put(None)

    async def translate_requests() -> None:
        while True:
            req = await queue.get()
            if req is None:
                break
            try:
                model = pick_model([], req.target_lang)
                async for delta in stream_translation(req.text, req.target_lang, model=model):
                    await _safe_send_json(
                        client_ws,
                        {
                            "type": "translation",
                            "segment_id": req.segment_id,
                            "data": delta,
                            "final": False,
                        },
                    )
                await _safe_send_json(
                    client_ws,
                    {
                        "type": "translation",
                        "segment_id": req.segment_id,
                        "data": "",
                        "final": True,
                    },
                )
            except Exception as exc:
                logger.exception("Translation failed for segment %s", req.segment_id)
                await _safe_send_json(
                    client_ws, {"type": "error", "data": f"Translation failed: {exc}"}
                )

    tasks = [
        asyncio.create_task(receive_requests(), name="receive_requests"),
        asyncio.create_task(translate_requests(), name="translate_requests"),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _safe_close(client_ws)
        logger.info("Translate session cleaned up")


async def _safe_send_json(ws: WebSocket, payload: dict) -> None:
    """Swallows send errors that happen after the client has already gone away."""
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def _safe_close(ws: WebSocket) -> None:
    """Closes the socket, tolerating a client that already disconnected or a
    close that was already sent (Starlette raises RuntimeError on double-close)."""
    try:
        await ws.close()
    except Exception:
        pass
