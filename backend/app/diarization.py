"""Streaming speaker diarization via diart (pyannote.audio-based), adapted so
a FastAPI websocket can push PCM16 chunks in instead of diart reading from a
mic or file.

diart/torch are optional, heavy dependencies (see requirements.txt) and are
imported lazily here, so a backend without them installed still serves
/ws/stream and /ws/translate normally; only /ws/diarize fails, with a clear
DiarizationUnavailable error, until they're installed and HUGGINGFACE_TOKEN
is set.

NOTE: diart's public API has moved around across versions (StreamingInference
construction args, SpeakerDiarizationConfig fields). The construction below
is defensive — it tries the tuned kwargs first and falls back to plain
defaults on TypeError. This has been exercised end-to-end (pipeline build,
model download, real audio) on Windows / Python 3.12 with the versions
pinned in requirements.txt; see that file's comments and
_patch_speechbrain_windows_lazy_import_bug() below for the specific
version-compatibility issues that came up and how each was resolved.
"""

import asyncio
import logging
import os
import threading
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger("babel.diarization")

SAMPLE_RATE = 16000


class DiarizationUnavailable(RuntimeError):
    """Raised when diart/torch aren't installed, no HF token is configured,
    or the pipeline otherwise can't be built."""


def _patch_speechbrain_windows_lazy_import_bug() -> None:
    """Works around a real cross-platform bug in speechbrain (pulled in by
    pyannote.audio) that only manifests on Windows.

    pytorch_lightning's checkpoint loader calls inspect.stack() to check for
    JIT scripting, which walks every frame on the call stack. If any of
    those frames happens to belong to one of speechbrain's lazy-loaded
    optional-integration proxy modules (e.g. its k2 or transformers
    integrations), that walk incidentally triggers the proxy's __getattr__,
    which tries to actually import the (possibly-not-installed) target.

    speechbrain's own LazyModule.ensure_module already guards against
    exactly this: it's supposed to detect that the import was triggered by
    `inspect.py` itself and raise AttributeError instead (so hasattr()
    checks upstream correctly see "not available" rather than crashing).
    But the guard checks `importer_frame.filename.endswith("/inspect.py")`
    -- a Unix-only path separator assumption -- so it silently never fires
    on Windows, and a plain ImportError (e.g. "please install k2") propagates
    out of a hasattr() call and crashes pipeline construction entirely, long
    before any actual k2/transformers functionality would be used.

    This replaces the method with a copy that checks the filename in a
    platform-independent way. Safe to call more than once; safe to call if
    speechbrain isn't installed (no-ops until diart/torch/speechbrain are).
    """
    try:
        from speechbrain.utils import importutils
    except ImportError:
        return

    import importlib
    import inspect
    import sys

    def patched_ensure_module(self, stacklevel: int):
        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except AttributeError:
            pass

        if importer_frame is not None and os.path.basename(importer_frame.filename) == "inspect.py":
            raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(f".{self.target}", self.package)
            except Exception as e:
                raise ImportError(f"Lazy import of {self!r} failed") from e

        return self.lazy_module

    importutils.LazyModule.ensure_module = patched_ensure_module


def _build_push_source(uri: str, sample_rate: int):
    """Returns a diart AudioSource that receives externally-pushed PCM16
    audio via push() instead of pulling from a mic/file. read() blocks (like
    diart's own MicrophoneAudioSource) until close() is called, keeping
    StreamingInference's run loop alive while push() feeds it from outside."""
    from diart.sources import AudioSource

    class PushAudioSource(AudioSource):
        def __init__(self) -> None:
            super().__init__(uri, sample_rate)
            self._closed = threading.Event()

        def read(self) -> None:
            self._closed.wait()

        def push(self, chunk: np.ndarray) -> None:
            if not self._closed.is_set():
                self.stream.on_next(chunk)

        def close(self) -> None:
            self.stream.on_completed()
            self._closed.set()

    return PushAudioSource()


def _build_pipeline():
    """Builds a diart SpeakerDiarization pipeline, tuned for low-latency
    live use (5s analysis window, 500ms shift, minimum-latency mode)."""
    from diart import SpeakerDiarization, SpeakerDiarizationConfig

    try:
        config = SpeakerDiarizationConfig(duration=5.0, step=0.5, latency="min")
    except TypeError:
        logger.warning(
            "Installed diart version doesn't accept the tuned SpeakerDiarizationConfig "
            "kwargs (duration/step/latency); falling back to its defaults."
        )
        config = SpeakerDiarizationConfig()
    return SpeakerDiarization(config)


class StreamingDiarizer:
    """Owns one diart streaming diarization pipeline for a single client
    session. Mirrors DeepgramLiveClient's shape: push_audio() feeds PCM16
    chunks in, turns() yields resolved (speaker, start_sec, end_sec) tuples
    out as diart resolves each analysis window."""

    def __init__(self, hf_token: str, max_speakers: int = 6):
        self._hf_token = hf_token
        self._max_speakers = max_speakers
        self._source = None
        self._inference = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._run_future = None
        self._queue: "asyncio.Queue[tuple[str, float, float] | None]" = asyncio.Queue()

    def start(self) -> None:
        if not self._hf_token:
            raise DiarizationUnavailable(
                "HUGGINGFACE_TOKEN is not set. diart's default pipeline needs two gated "
                "models: accept terms at https://huggingface.co/pyannote/segmentation and "
                "https://huggingface.co/pyannote/embedding, then set HUGGINGFACE_TOKEN in "
                "backend/.env."
            )
        # huggingface_hub resolves auth from these env vars regardless of the
        # exact diart/pyannote version's constructor signature, which is more
        # version-stable than passing a token through diart's own API.
        os.environ.setdefault("HF_TOKEN", self._hf_token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", self._hf_token)

        try:
            from diart.inference import StreamingInference
        except ImportError as exc:
            raise DiarizationUnavailable(
                "diart/torch are not installed; run `pip install diart torch`."
            ) from exc

        _patch_speechbrain_windows_lazy_import_bug()

        try:
            pipeline = _build_pipeline()
        except Exception as exc:
            raise DiarizationUnavailable(f"Failed to build diart pipeline: {exc}") from exc

        self._loop = asyncio.get_event_loop()
        self._source = _build_push_source("babel-client", SAMPLE_RATE)

        try:
            self._inference = StreamingInference(pipeline, self._source, do_plot=False, show_progress=False)
        except TypeError:
            self._inference = StreamingInference(pipeline, self._source)
        self._inference.attach_hooks(self._on_prediction)

        # StreamingInference has no .run() -- it's invoked via __call__ (see
        # diart's own console/stream.py). It blocks pumping the (blocking)
        # source, so it must live on a worker thread; results flow back via
        # the queue populated by _on_prediction.
        self._run_future = self._loop.run_in_executor(None, self._inference)

    def _on_prediction(self, prediction) -> None:
        try:
            annotation = prediction[0] if isinstance(prediction, tuple) else prediction
            for segment, _track, speaker in annotation.itertracks(yield_label=True):
                item = (str(speaker), float(segment.start), float(segment.end))
                self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except Exception:
            logger.exception("Error handling diart prediction")

    async def push_audio(self, chunk: bytes) -> None:
        if self._source is None:
            raise RuntimeError("StreamingDiarizer.start() must be called before pushing audio")
        pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self._source.push(pcm.reshape(1, -1))  # diart expects (channels, samples)

    async def turns(self) -> AsyncIterator[tuple[str, float, float]]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        if self._source is not None:
            self._source.close()
        if self._run_future is not None:
            try:
                await asyncio.wait_for(self._run_future, timeout=5)
            except Exception:
                pass
        await self._queue.put(None)
