# Babel — Real-Time Audio Translator

Low-latency speech-to-speech captioning. Two clients share one backend:

- **Web / PWA** (`frontend/`) — **the primary way to use Babel on a phone.** Mic audio → Deepgram Nova-3 (streaming STT, with automatic language switching for bilingual/multilingual speech) → Groq Llama (streaming translation) → browser, over `/ws/stream`, plus an optional second connection to `/ws/diarize` for speaker separation. Installable to an iPhone home screen (manifest + service worker), no app store, no Mac required. See "Run" below for setup and "Test locally" for the on-phone install steps.
- **iOS native app** (`ios/`) — a secondary/reference path: a SwiftUI app using Apple's on-device Speech and Translation frameworks instead of the cloud pipeline. **Requires Xcode on macOS to build** (or a cloud Mac/CI service) — see [ios/README.md](ios/README.md). Not the recommended path unless you have Mac access, since the PWA gets you running today with no build step at all.

## Layout

```
Babel/
  backend/
    app/
      main.py             FastAPI app: /ws/stream, /ws/diarize, /ws/translate
      deepgram_client.py  Deepgram live-transcription WebSocket wrapper
      translator.py       Groq streaming translation
      diarization.py      diart (pyannote) streaming speaker diarization
      config.py           Env-var settings
      schemas.py          WebSocket message shapes
    scripts/
      test_translate.py   Manual smoke test for /ws/translate
      test_diarize.py     Manual smoke test for /ws/diarize
    requirements.txt
    .env.example
  frontend/                PWA: the primary phone client
    index.html            Mic capture, bubble-feed UI, settings panel
    manifest.webmanifest  PWA manifest (installable to iOS home screen)
    sw.js                 Service worker (app-shell caching)
    icons/                Placeholder PNG icons (swap for real branding)
  ios/                     Native app: secondary/reference path, needs a Mac
    Babel/                SwiftUI app source (see ios/README.md)
    project.yml            XcodeGen project manifest
  README.md
```

## How it works

1. The client opens `wss://<host>/ws/stream?lang=<target>&source_langs=<code1,code2,...>` and starts streaming raw 16kHz mono PCM16 audio chunks (~50ms each), captured via an `AudioWorklet` in the browser (falls back to `ScriptProcessorNode` if unavailable).
2. FastAPI accepts the client connection and decides how to talk to Deepgram based on the candidate languages (see "Language switching" below): either one `language=multi` connection that auto-switches natively, or one single-language connection that the client can manually re-point.
3. Deepgram streams back interim and final transcripts. Interim results (`is_final=false`) are relayed to the client immediately, untranslated, for instant visual feedback. The server replies with `{"type": "status", "data": "connected", "multi_mode": bool, "language": "<code>"}` right after connecting, so the client knows whether to show a manual language picker.
4. Each final segment is pushed onto an internal queue and translated by a dedicated background worker calling Groq with `stream=True`. Translations for different segments are processed in order (so they don't interleave), while STT keeps flowing independently in the meantime.
5. All server→client messages are structured JSON: `{"type": "transcript", "segment_id": N, "data": "...", "is_final": bool, "language": "<code>"}` and `{"type": "translation", "segment_id": N, "data": "...", "final": bool}` (translation arrives as a stream of deltas, terminated by one `final: true` message with empty `data`).
6. If speaker separation is turned on, the client opens a second connection to `/ws/diarize` with the same audio and merges the resulting `speaker_turn` events into the transcript by timestamp, client-side.

### Language switching

Bilingual/multilingual sessions (`source_langs` has more than one code) are handled one of two ways, decided once at connect time by `MULTI_MODE_LANGUAGES` in `app/main.py`:

- **Every candidate language is in Deepgram's native code-switching set** (`en`, `es`, `fr`, `de`, `hi`, `ru`, `pt`, `ja`, `it`, `nl` for Nova-3, checked 2026-08) — the whole session connects with `language=multi`, and Deepgram itself detects and transcribes each language correctly without any reconnection. Each transcript message's `language` field comes straight from Deepgram's own per-result `channel.alternatives[0].languages`.
- **Any candidate language falls outside that set** (Arabic, Turkish, Tagalog, Tamil, ...) — no automatic switching is attempted for the whole session, even for the covered languages in a mixed set. Instead the client sends a `{"type": "switch_language", "language": "<code>"}` text frame whenever the user manually picks a different language, and the server reconnects Deepgram with the new language, replying with `{"type": "status", "data": "language_switched", "language": "<code>"}` once it's live.

An earlier version of this attempted automatic switching for *any* language combination by running finalized Deepgram transcripts through `langid.py` and reconnecting on a detected shift. That doesn't work reliably: if the wrong-language connection is still active when a speaker switches, it transcribes the new language using the old one's model, producing garbled text that still reads as the old language to a text classifier — so the switch can never be detected. Native `multi` mode doesn't have this chicken-and-egg problem since Deepgram listens for all its covered languages simultaneously; outside that set there's no substitute, hence the manual picker.

## `/ws/diarize` and `/ws/translate`

Originally built for the iOS app, both are also usable by (and in the PWA's
case, actively used by) the web client:

- **`/ws/diarize`** — accepts a raw PCM16 audio stream (same format as
  `/ws/stream`) and streams back `{"type": "speaker_turn", "speaker": "S1",
  "start": 12.3, "end": 15.1}` events as `diart` resolves each ~5s analysis
  window. Independent of `/ws/stream`, so a client can run STT/translation and
  diarization concurrently over the same audio. See `app/diarization.py` for
  implementation notes and version caveats. The PWA's diarization toggle
  (off by default) controls whether it opens this connection at all.
- **`/ws/translate`** — accepts `{"segment_id", "text", "target_lang"}` JSON
  requests and streams back translation deltas (same message shape as
  `/ws/stream`'s translation messages), reusing `translator.py` unchanged.
  Used by the iOS app when on-device STT already produced text but Apple's
  on-device `Translation` framework doesn't support the language pair; the
  PWA doesn't need this since `/ws/stream` already returns translated text.

Before wiring a client to a new backend, sanity-check these two directly:

```bash
cd backend
python scripts/test_translate.py "Hello, how are you?" Spanish
python scripts/test_diarize.py path/to/16k_mono.wav
```

## Setup

**Requirements:** Python 3.11 or 3.12 (**not 3.13** — see below), a Deepgram
API key, a Groq API key, and (for `/ws/diarize`) a Hugging Face token.

```bash
cd backend
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `backend/.env` and fill in `DEEPGRAM_API_KEY` and `GROQ_API_KEY`. For
diarization, also accept the gated models' terms at
https://huggingface.co/pyannote/segmentation **and**
https://huggingface.co/pyannote/embedding (diart's actual default pipeline —
not `pyannote/speaker-diarization-3.1`, despite that being the model most
pyannote docs point you to), generate a token at
https://huggingface.co/settings/tokens, and set `HUGGINGFACE_TOKEN`. The
`diart`/`torch` dependencies are heavy (first run downloads pretrained
models) and only `/ws/diarize` needs them — the rest of the backend works
without them installed.

**Why Python 3.12, not 3.13:** `diart` pins `numpy<2.0`, and numpy stopped
publishing prebuilt wheels for the 1.x line before Python 3.13 existed —
installing on 3.13 tries to compile numpy from source and fails without a C
compiler. `requirements.txt` also pins `torch`/`torchaudio`/`torchvision`,
`matplotlib`, and `huggingface_hub` to specific ranges beyond what diart
itself requires, because their latest releases have each individually
broken something pyannote.audio 3.4.0 still relies on (removed
`torchaudio.AudioMetaData`, removed `matplotlib.cm.get_cmap`, removed the
`use_auth_token` kwarg on `hf_hub_download` respectively) — see the comments
in `requirements.txt` if a future `pip install` starts failing again in this
area.

## Run

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000/static/index.html** in a desktop browser (needs `getUserMedia` + `AudioWorklet` support, and a secure context — `localhost` counts) to test on this machine.

The backend also serves the frontend directly via `/static`, so no separate static server is needed. On a phone, the frontend's Settings panel (gear icon) lets you point at a backend running elsewhere (e.g. your dev machine's LAN IP) without editing any files — see "On your phone" below.

## Test locally (desktop)

1. Start the server as above and open the page.
2. Tap the gear icon, pick a target language and the language(s) you'll be speaking, close Settings.
3. Click **Start Listening**, grant microphone permission.
4. Speak — a bubble appears per finalized segment, with the source text filling in first and the translation shortly after (translating… while it's in flight). An italic line above the bubbles shows the current interim (not-yet-final) transcript.
5. Click **Stop Listening** to close the mic and WebSocket(s) cleanly.
6. `GET /health` returns `{"status": "ok"}` for a quick backend liveness check.

## Test on your phone

1. Make sure your phone and the machine running the backend are on the same network, and find that machine's LAN IP (e.g. `ipconfig getifaddr en0` on a Mac, `ipconfig` on Windows).
2. On the backend machine, run `uvicorn app.main:app --host 0.0.0.0 --port 8000` (note `--host 0.0.0.0` — the default `127.0.0.1` only accepts connections from the same machine) and make sure the OS firewall allows inbound connections on port 8000.
3. On the phone, open Safari to `http://<LAN-IP>:8000/static/index.html`.
4. Tap **Share → Add to Home Screen** (the app shows a one-time banner reminding you of this, since Safari has no automatic install prompt). Launching from the home screen icon runs it in standalone mode (no browser chrome) and keeps the screen awake while listening (Screen Wake Lock API).
5. In Settings, the Backend field can stay blank (same-origin) since you loaded the page directly from the backend's address.
6. Everything else matches the desktop flow above. Speaker separation needs `HUGGINGFACE_TOKEN`/`diart` configured on the backend (see Setup) — the toggle is off by default so the rest of the app works without that setup.

### Sanity-checking the pipeline pieces independently

- Deepgram connectivity: bad/missing `DEEPGRAM_API_KEY` surfaces as a `{"type": "error", ...}` message immediately on connect (visible in the status pill and browser console).
- Groq connectivity: a translation failure surfaces as `{"type": "error", "data": "Translation failed: ..."}` without tearing down the transcript stream.
- Watch server logs (`uvicorn` stdout) — each stage logs on failure with a full traceback.

## Notes on the design choices

- **Why a translation queue instead of `asyncio.create_task` per segment:** translating segments fully in parallel would let a slow Groq response for segment 1 finish after segment 2, causing translations to render out of order. A single background worker consuming a queue keeps translations in the order they were spoken while still running fully decoupled from — and non-blocking of — the Deepgram forwarding loop.
- **Why stride-decimation resampling in the worklet:** it's cheap enough to run per-sample in the audio thread and is sufficient quality for speech STT; it intentionally skips an anti-aliasing filter for simplicity. If you see STT quality issues, that's the first place to upgrade (e.g. a proper polyphase resampler).
- **Model choice:** `GROQ_MODEL` defaults to `llama-3.1-70b-versatile`; set it to a spec-decoded variant (e.g. `llama-3.3-70b-specdec`) in `.env` for lower per-token latency if your Groq account has access.
