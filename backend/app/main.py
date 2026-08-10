"""FastAPI orchestrator: client audio <-> Deepgram STT <-> Groq translation."""

import asyncio
import itertools
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import settings
from .deepgram_client import DeepgramLiveClient
from .diarization import DiarizationUnavailable, StreamingDiarizer
from .schemas import TranslateRequest
from .translator import stream_translation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("babel.main")

# Deepgram's live API has no auto-detection for a single-language connection,
# so the source language must be picked explicitly and validated (it flows
# into an outbound API call). Nova-3 language codes; see
# https://developers.deepgram.com/docs/models-languages-overview (checked
# 2026-08). Malayalam ("ml") is deliberately excluded -- confirmed absent
# from Deepgram's supported list for both Nova-2 and Nova-3. "tl" (Tagalog)
# similarly isn't documented but does work in practice.
SUPPORTED_SOURCE_LANGUAGES = {
    "en", "es", "fr", "de", "zh", "ja", "ko", "pt", "hi", "ar", "tr", "tl", "ta",
}

# Deepgram's live `language=multi` mode transcribes multiple languages
# within one stream without needing to know in advance which one is coming
# (see https://developers.deepgram.com/docs/multilingual-code-switching,
# checked 2026-08) -- but it only covers this fixed set for Nova-3. Earlier
# attempts at auto-switching by reconnecting Deepgram based on langid-ing
# its own output text were unreliable: if the wrong-language connection is
# still active when a speaker switches, it transcribes the new language
# using the old one's model, producing garbled text that still reads as the
# old language to a text classifier -- the switch can never be detected.
# Native multi mode doesn't have this chicken-and-egg problem since Deepgram
# itself listens for all covered languages simultaneously. For anything
# outside this set (Arabic, Turkish, Tagalog, Tamil, ...), there's no
# reliable automatic option, so the client should offer a manual language
# picker instead (see the "switch_language" control message below).
MULTI_MODE_LANGUAGES = {"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"}

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


@app.get("/health")
async def health():
    return {"status": "ok"}


def _parse_candidate_languages(client_ws: WebSocket) -> list[str]:
    """Reads ?source_langs=en,es (new, multi-candidate) or falls back to the
    older single ?source_lang=en for backward compatibility."""
    multi = client_ws.query_params.get("source_langs")
    if multi:
        candidates = [c.strip() for c in multi.split(",") if c.strip() in SUPPORTED_SOURCE_LANGUAGES]
        if candidates:
            return candidates

    single = client_ws.query_params.get("source_lang", settings.source_language)
    return [single if single in SUPPORTED_SOURCE_LANGUAGES else settings.source_language]


@app.websocket("/ws/stream")
async def stream_endpoint(client_ws: WebSocket):
    await client_ws.accept()

    target_language = client_ws.query_params.get("lang", settings.target_language)
    candidate_langs = _parse_candidate_languages(client_ws)
    current_lang = candidate_langs[0]

    # Native code-switching only when every candidate is in Deepgram's
    # covered set (see MULTI_MODE_LANGUAGES above); otherwise this session
    # has no automatic switching and relies on the client sending explicit
    # "switch_language" control messages (e.g. from a manual picker in the UI).
    uses_multi_mode = len(candidate_langs) > 1 and all(lang in MULTI_MODE_LANGUAGES for lang in candidate_langs)
    deepgram_language = "multi" if uses_multi_mode else current_lang

    dg_client = DeepgramLiveClient(
        settings.deepgram_api_key, model=settings.deepgram_stt_model, language=deepgram_language
    )
    try:
        await dg_client.connect()
    except Exception as exc:
        logger.exception("Failed to connect to Deepgram")
        await _safe_send_json(client_ws, {"type": "error", "data": f"Could not connect to speech engine: {exc}"})
        await _safe_close(client_ws)
        return

    await _safe_send_json(
        client_ws,
        {"type": "status", "data": "connected", "multi_mode": uses_multi_mode, "language": current_lang},
    )

    segment_ids = itertools.count(1)
    translation_queue: "asyncio.Queue[tuple[int, str] | None]" = asyncio.Queue()
    dg_lock = asyncio.Lock()
    session_ended = asyncio.Event()
    deepgram_reader_task: Optional[asyncio.Task] = None

    async def run_deepgram_reader(client: DeepgramLiveClient) -> None:
        """Reads transcript messages from one Deepgram connection. Cancelled
        (not left to finish naturally) when switch_to_language() swaps in a
        new connection; only an unexpected Deepgram-side failure ends the
        session from here."""
        nonlocal current_lang
        try:
            async for message in client.messages():
                alternatives = message.get("channel", {}).get("alternatives", [{}])
                text = alternatives[0].get("transcript", "") if alternatives else ""
                if not text:
                    continue

                is_final = bool(message.get("is_final", False))
                seg_id = next(segment_ids) if is_final else -1

                # In multi mode Deepgram reports the detected language per
                # result (channel.alternatives[0].languages, a list of
                # BCP-47 tags); otherwise it's whatever we're connected with.
                if uses_multi_mode:
                    detected = alternatives[0].get("languages") if alternatives else None
                    segment_language = detected[0] if detected else current_lang
                else:
                    segment_language = current_lang

                await _safe_send_json(
                    client_ws,
                    {
                        "type": "transcript",
                        "segment_id": seg_id,
                        "data": text,
                        "is_final": is_final,
                        "language": segment_language,
                    },
                )

                if is_final:
                    await translation_queue.put((seg_id, text))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error reading from Deepgram")
            session_ended.set()

    async def switch_to_language(new_lang: str) -> None:
        """Manual switch requested by the client (see handle_control_message).
        Deepgram has no in-place way to change a live connection's language,
        so this opens a new connection and swaps it in."""
        nonlocal dg_client, current_lang, deepgram_reader_task
        if uses_multi_mode:
            return  # this session auto-switches via Deepgram's own multi mode
        if new_lang not in candidate_langs or new_lang == current_lang:
            return

        try:
            new_client = DeepgramLiveClient(
                settings.deepgram_api_key, model=settings.deepgram_stt_model, language=new_lang
            )
            await new_client.connect()
        except Exception as exc:
            logger.exception("Failed to switch source language to %s", new_lang)
            await _safe_send_json(client_ws, {"type": "error", "data": f"Could not switch to {new_lang}: {exc}"})
            return

        old_client = dg_client
        async with dg_lock:
            dg_client = new_client
            current_lang = new_lang
        if deepgram_reader_task is not None:
            deepgram_reader_task.cancel()
        await old_client.close()
        await _safe_send_json(client_ws, {"type": "status", "data": "language_switched", "language": new_lang})
        deepgram_reader_task = asyncio.create_task(run_deepgram_reader(new_client), name="deepgram_reader")

    async def handle_control_message(raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if payload.get("type") == "switch_language":
            await switch_to_language(payload.get("language", ""))

    async def client_reader() -> None:
        """Reads raw client WebSocket frames: binary frames are audio,
        forwarded to whichever Deepgram connection is currently active; text
        frames are JSON control messages (currently just switch_language)."""
        try:
            while True:
                message = await client_ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    async with dg_lock:
                        client = dg_client
                    try:
                        await client.send_audio(data)
                    except Exception:
                        # Most likely landed exactly during a language-switch
                        # reconnect; drop this ~50ms chunk rather than
                        # tearing down the whole forwarding loop over it.
                        pass
                    continue
                text = message.get("text")
                if text is not None:
                    await handle_control_message(text)
        except WebSocketDisconnect:
            logger.info("Client disconnected (audio stream)")
        except Exception:
            logger.exception("Error in client reader")
        finally:
            session_ended.set()

    async def translation_worker() -> None:
        while True:
            item = await translation_queue.get()
            if item is None:
                break
            seg_id, text = item
            try:
                async for delta in stream_translation(text, target_language):
                    await _safe_send_json(
                        client_ws,
                        {"type": "translation", "segment_id": seg_id, "data": delta, "final": False},
                    )
                await _safe_send_json(
                    client_ws,
                    {"type": "translation", "segment_id": seg_id, "data": "", "final": True},
                )
            except Exception as exc:
                logger.exception("Translation failed for segment %s", seg_id)
                await _safe_send_json(client_ws, {"type": "error", "data": f"Translation failed: {exc}"})

    deepgram_reader_task = asyncio.create_task(run_deepgram_reader(dg_client), name="deepgram_reader")
    client_reader_task = asyncio.create_task(client_reader(), name="client_reader")
    translation_task = asyncio.create_task(translation_worker(), name="translation_worker")

    try:
        await session_ended.wait()
    finally:
        client_reader_task.cancel()
        deepgram_reader_task.cancel()
        await translation_queue.put(None)
        await asyncio.gather(client_reader_task, deepgram_reader_task, translation_task, return_exceptions=True)
        await dg_client.close()
        await _safe_close(client_ws)
        logger.info("Session cleaned up")


@app.websocket("/ws/diarize")
async def diarize_endpoint(client_ws: WebSocket):
    """Accepts a raw PCM16 audio stream (same format as /ws/stream) and
    streams back speaker-turn events as diart resolves each analysis window.
    Independent of /ws/stream so a client can run on-device STT and cloud
    diarization concurrently over the same audio."""
    await client_ws.accept()

    diarizer = StreamingDiarizer(
        hf_token=settings.huggingface_token,
        max_speakers=settings.diarization_max_speakers,
    )
    try:
        diarizer.start()
    except DiarizationUnavailable as exc:
        logger.error("Diarization unavailable: %s", exc)
        await _safe_send_json(client_ws, {"type": "error", "data": str(exc)})
        await _safe_close(client_ws)
        return

    await _safe_send_json(client_ws, {"type": "status", "data": "connected"})

    async def client_to_diarizer() -> None:
        try:
            while True:
                chunk = await client_ws.receive_bytes()
                await diarizer.push_audio(chunk)
        except WebSocketDisconnect:
            logger.info("Client disconnected (diarize stream)")
        except Exception:
            logger.exception("Error forwarding audio to diarizer")

    async def diarizer_to_client() -> None:
        try:
            async for speaker, start, end in diarizer.turns():
                await _safe_send_json(
                    client_ws,
                    {"type": "speaker_turn", "speaker": speaker, "start": start, "end": end},
                )
        except Exception:
            logger.exception("Error reading from diarizer")

    tasks = [
        asyncio.create_task(client_to_diarizer(), name="client_to_diarizer"),
        asyncio.create_task(diarizer_to_client(), name="diarizer_to_client"),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await diarizer.close()
        await _safe_close(client_ws)
        logger.info("Diarization session cleaned up")


@app.websocket("/ws/translate")
async def translate_endpoint(client_ws: WebSocket):
    """Accepts {"segment_id", "text", "target_lang"} JSON requests and streams
    back translation deltas, reusing stream_translation(). Decoupled from STT
    so a client that already has transcribed text (e.g. on-device) can get a
    cloud translation fallback without re-uploading audio. A single worker
    processes requests in order, so responses can't interleave out of order
    the way concurrent tasks per segment would."""
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
                    await _safe_send_json(client_ws, {"type": "error", "data": f"Invalid request: {exc}"})
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
                async for delta in stream_translation(req.text, req.target_lang):
                    await _safe_send_json(
                        client_ws,
                        {"type": "translation", "segment_id": req.segment_id, "data": delta, "final": False},
                    )
                await _safe_send_json(
                    client_ws,
                    {"type": "translation", "segment_id": req.segment_id, "data": "", "final": True},
                )
            except Exception as exc:
                logger.exception("Translation failed for segment %s", req.segment_id)
                await _safe_send_json(client_ws, {"type": "error", "data": f"Translation failed: {exc}"})

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
