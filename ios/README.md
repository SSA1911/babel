# Babel iOS — native hybrid client

> **This is a secondary/reference path.** Building it requires Xcode on
> macOS (or a cloud Mac/CI service) — if you don't have Mac access, use the
> PWA in [`../frontend/`](../frontend/index.html) instead, which now covers
> the same live transcription/translation/diarization/language-switching
> goals through the browser, installable to an iPhone home screen with no
> build step. See the root [README.md](../README.md) for how the two
> compare.

A SwiftUI app that transcribes and translates live conversation, preferring
Apple's on-device Speech and Translation frameworks and only calling the
FastAPI backend (`../backend/`) when Apple's tools can't do the job:
speaker diarization always (Apple has no on-device diarization API), and
STT/translation as fallbacks when a needed language isn't available
on-device. See the root [README.md](../README.md) and
`backend/app/diarization.py` for the backend side of this.

**I wrote this on Windows without Xcode or a Swift toolchain, so none of it
has been compiled.** The code is written carefully against documented
iOS 17.4+ APIs, but you're the first compiler it'll see — budget time for
build fixes, especially around the Translation framework and diart's exact
API surface on the backend (both are flagged in code comments where I was
least certain).

## Setup

1. Install Xcode (from the Mac App Store) and [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).
2. From this `ios/` directory: `xcodegen generate` — this reads `project.yml` and produces `Babel.xcodeproj`, so there's no hand-maintained project file to go stale.
3. Open `Babel.xcodeproj` in Xcode, pick a development team under the target's Signing & Capabilities tab.
4. Run on a **physical device**, not the simulator — on-device Speech recognition and Translation asset downloads need real hardware to behave realistically (the simulator's mic/on-device-model support is unreliable for this).
5. On first launch, grant microphone and speech recognition permissions when prompted.
6. Open the gear icon → Settings, and:
   - Point **Backend** at your running FastAPI server (e.g. `ws://192.168.1.23:8000` — your Mac's LAN IP if testing against `uvicorn --reload` locally; iOS can't reach `localhost` meaning the phone itself). Get your Mac's LAN IP with `ipconfig getifaddr en0`.
   - Pick a **target language**.
   - Pick every **spoken language** you expect to hear (one for monolingual use, two or more for bilingual/multilingual auto-switching).
7. Make sure the backend from `../backend/` is running and reachable from the phone (same Wi-Fi network as your Mac, and the Mac's firewall allows inbound connections on port 8000).

## Architecture

```
AudioCaptureEngine (AVAudioEngine)
  ├─ native-format buffers  → SpeechRecognitionManager (SFSpeechRecognizer, on-device)
  │                              ├─ interim/final text → LanguageSwitchDetector (NLLanguageRecognizer)
  │                              │     → if language changed: SpeechRecognitionManager.switchLocale(to:)
  │                              └─ final segments      → TranslationManager (on-device Translation,
  │                                                         falls back to backend /ws/translate)
  └─ resampled 16kHz PCM16   → DiarizationClient (backend /ws/diarize, always)
                              → FallbackStreamClient (backend /ws/stream, only if no on-device
                                                        recognizer covers a configured language)

ConversationSession merges TranscriptSegments + SpeakerTurns by timestamp → published [ConversationTurn] → ConversationView
```

**STT path is chosen once per session, not per segment.** At the start of
`ConversationSession.start()`, `SpeechRecognitionManager.supportsOnDevice(locales:)`
checks every configured spoken language for an on-device-capable
`SFSpeechRecognizer`. If all are covered, the whole session runs on-device
(with automatic language switching). If any aren't, the whole session uses
the `/ws/stream` cloud fallback instead — including for languages that *do*
have on-device support, since Deepgram (the backend's cloud STT) only
accepts one fixed source language per connection with no auto-detect, so
there's no way to mix on-device and cloud STT language-by-language within
one session.

**Diarization is always backend-based**, streamed on its own `/ws/diarize`
connection regardless of which STT path is active, since Apple has no
public on-device diarization API. Speaker turns resolve roughly 5 seconds
behind the audio (diart's analysis window) and are merged into the
transcript by matching each `ConversationTurn`'s timestamp against the
speaker turns' `[start, end]` ranges (`ConversationSession.applySpeakerLabels`)
— so a speaker label can visibly attach to a bubble a moment after its text
appears. That's diart resolving its window, not a bug.

## Known limitations (by design, not oversights)

- **Bilingual/multilingual auto-switching only works on-device.** The cloud
  fallback path is stuck on one language for the whole session (a Deepgram
  API constraint, documented in `backend/app/main.py`).
- **Language switching has a brief gap.** `SFSpeechRecognizer` has no
  code-switching support — switching languages means canceling the current
  recognition task and starting a new one with a different locale, which
  drops whatever was mid-utterance at that exact moment.
- **Speaker labels lag transcript text** by up to ~5s (diart's window) and,
  rarely, a turn near a true speaker change might get attributed to the
  wrong neighboring speaker (the point-based nearest-turn fallback in
  `applySpeakerLabels`).
- **Diarization needs a `HUGGINGFACE_TOKEN`** on the backend (gated
  pyannote model) — see `backend/README.md`. Without it, `/ws/diarize`
  reports an error and every turn stays unlabeled; transcription and
  translation are unaffected.

## Manual test plan

Since there's no automated test target here (and I can't run one), verify
in this order once you have a build running on a device:

1. **Backend sanity first** — before touching the app, run
   `backend/scripts/test_translate.py` and `backend/scripts/test_diarize.py`
   (see root README) against your running backend. Fix backend issues here
   before blaming the app.
2. **Single speaker, single language.** Set one spoken language matching
   what you'll say, pick any target. Start, speak a few sentences, confirm
   transcript + translation both appear and the status pill reads
   "listening (on-device)". Every turn should end up with the same
   (or no, until diarization catches up) speaker label.
3. **Two speakers, one language.** Same settings, two people alternating.
   Confirm distinct speaker labels/colors appear once diarization catches
   up (give it several seconds after each turn starts).
4. **Bilingual, one speaker.** Select two spoken languages you can both
   speak. Say a sentence in language A, then B, then A again. Confirm the
   status stays "listening (on-device)" (if it instead falls back to cloud,
   your device likely lacks an on-device recognizer for one of the two —
   check by removing one language and retrying), and that the language tag
   on each bubble changes correctly. Expect a short gap right at each
   switch point.
5. **Two speakers, two languages (the real target scenario).** Combine 3
   and 4. Confirm both speaker separation and language tags update
   correctly turn-by-turn.
6. **Forced cloud fallback.** Pick a spoken language your device has no
   on-device recognizer for (or toggle Airplane Mode off but disable
   on-device Speech temporarily some other way if you can't find one) and
   confirm the status pill reads "listening (cloud fallback)" and
   transcript/translation still arrive (now via Deepgram+Groq).
