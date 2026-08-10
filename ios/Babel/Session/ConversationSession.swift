import AVFoundation
import Foundation

/// Top-level orchestrator: starts/stops audio capture, picks on-device vs.
/// cloud-fallback STT for the session, runs language-switch detection,
/// drives translation, and merges diarization speaker turns into the
/// published transcript by timestamp. This is the piece that ties every
/// other component together; see ios/README.md for the full data-flow
/// description and the trade-offs baked into the merge logic below.
@MainActor
final class ConversationSession: ObservableObject {
    @Published var turns: [ConversationTurn] = []
    @Published var interimText = ""
    @Published var interimLocale: Locale?
    @Published var isRunning = false
    @Published var statusMessage = "idle"
    @Published var lastError: String?

    let settings: AppSettings
    let backendConfig: BackendConfig
    let translationManager: TranslationManager

    private var audioEngine: AudioCaptureEngine?
    private var speechManager: SpeechRecognitionManager?
    private var languageDetector: LanguageSwitchDetector?
    private var fallbackClient: FallbackStreamClient?
    private var diarizationClient: DiarizationClient?
    private var speakerTurns: [SpeakerTurn] = []

    init(settings: AppSettings, backendConfig: BackendConfig) {
        self.settings = settings
        self.backendConfig = backendConfig
        translationManager = TranslationManager(backendConfig: backendConfig)
    }

    func start() async {
        guard !isRunning else { return }
        lastError = nil
        turns = []
        speakerTurns = []
        statusMessage = "requesting permissions…"

        guard await requestMicPermission() else {
            lastError = "Microphone permission denied. Enable it in iOS Settings > Babel."
            statusMessage = "idle"
            return
        }

        let candidateLocales = settings.candidateLanguages.map(\.locale)
        let speechAuthorized = await SpeechRecognitionManager.requestAuthorization()
        let onDeviceCapable = SpeechRecognitionManager.supportsOnDevice(locales: candidateLocales)
        let usingOnDeviceSTT = speechAuthorized && onDeviceCapable

        let backendClient = BackendClient(config: backendConfig)

        let diarization = DiarizationClient(client: backendClient)
        diarization.onError = { [weak self] message in
            Task { @MainActor in self?.lastError = message }
        }
        diarizationClient = diarization
        diarization.start()
        consumeSpeakerTurns()

        let engine = AudioCaptureEngine()
        audioEngine = engine

        if usingOnDeviceSTT {
            statusMessage = "listening (on-device)"
            let initialLanguage = settings.candidateLanguages[0]
            let manager = SpeechRecognitionManager(initialLocale: initialLanguage.locale)
            let detector = LanguageSwitchDetector(candidateLocales: candidateLocales)
            speechManager = manager
            languageDetector = detector
            translationManager.configure(source: initialLanguage.language, target: settings.targetLanguage.language)

            engine.onNativeBuffer = { [weak manager] buffer in manager?.append(buffer) }
            engine.onPCM16Chunk = { [weak diarization] data, _ in diarization?.push(data) }

            do {
                try engine.start()
            } catch {
                lastError = "Could not start audio: \(error.localizedDescription)"
                statusMessage = "idle"
                return
            }
            manager.start(sessionStartTime: engine.sessionStartTime ?? Date())
            consumeOnDeviceSegments(manager: manager, detector: detector)
        } else {
            // No on-device recognizer for one of the candidate languages (or
            // Speech permission was denied): use the whole-session cloud
            // pipeline instead. Deepgram has no auto-detect either, so this
            // path only ever uses settings.candidateLanguages[0] -- bilingual
            // switching is an on-device-only capability (see ios/README.md).
            statusMessage = "listening (cloud fallback)"
            let source = settings.candidateLanguages[0]
            let target = settings.targetLanguage
            let fallback = FallbackStreamClient(client: backendClient, source: source, target: target)
            fallbackClient = fallback
            fallback.start()

            engine.onPCM16Chunk = { [weak diarization, weak fallback] data, _ in
                diarization?.push(data)
                fallback?.push(data)
            }

            do {
                try engine.start()
            } catch {
                lastError = "Could not start audio: \(error.localizedDescription)"
                statusMessage = "idle"
                return
            }
            consumeFallbackEvents(fallback: fallback, sourceLocale: source.locale)
        }

        isRunning = true
    }

    func stop() {
        audioEngine?.stop()
        speechManager?.stop()
        fallbackClient?.stop()
        diarizationClient?.stop()
        translationManager.stop()
        audioEngine = nil
        speechManager = nil
        languageDetector = nil
        fallbackClient = nil
        diarizationClient = nil
        isRunning = false
        interimText = ""
        interimLocale = nil
        statusMessage = "stopped"
    }

    private func requestMicPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    private func consumeOnDeviceSegments(manager: SpeechRecognitionManager, detector: LanguageSwitchDetector) {
        Task {
            for await segment in manager.segments {
                if let newLocale = detector.evaluate(text: segment.text, currentLocale: manager.currentLocale) {
                    manager.switchLocale(to: newLocale)
                    if let newLanguage = SupportedLanguage.byLocaleIdentifier(newLocale.identifier) {
                        translationManager.configure(source: newLanguage.language, target: settings.targetLanguage.language)
                    }
                }

                if segment.isFinal {
                    interimText = ""
                    interimLocale = nil
                    appendOrUpdateTurn(id: segment.id, text: segment.text, locale: segment.locale, timestamp: segment.timestamp, isFinal: true)
                    translateOnDeviceSegment(id: segment.id, text: segment.text, locale: segment.locale)
                } else {
                    interimText = segment.text
                    interimLocale = segment.locale
                }
            }
        }
    }

    private func consumeFallbackEvents(fallback: FallbackStreamClient, sourceLocale: Locale) {
        Task {
            for await event in fallback.events {
                switch event {
                case .transcript(let segment):
                    if segment.isFinal {
                        interimText = ""
                        interimLocale = nil
                        let elapsed = audioEngine?.sessionStartTime.map { Date().timeIntervalSince($0) } ?? 0
                        appendOrUpdateTurn(id: segment.id, text: segment.text, locale: sourceLocale, timestamp: elapsed, isFinal: true)
                    } else {
                        interimText = segment.text
                        interimLocale = sourceLocale
                    }
                case .translationDelta(let segmentID, let text, let isFinal):
                    applyTranslationDelta(segmentID: segmentID, delta: text, isFinal: isFinal)
                case .error(let message):
                    lastError = message
                }
            }
        }
    }

    private func consumeSpeakerTurns() {
        guard let diarizationClient else { return }
        Task {
            for await turn in diarizationClient.turns {
                speakerTurns.append(turn)
                applySpeakerLabels()
            }
        }
    }

    private func appendOrUpdateTurn(id: Int, text: String, locale: Locale, timestamp: TimeInterval, isFinal: Bool) {
        if let index = turns.firstIndex(where: { $0.id == id }) {
            turns[index].sourceText = text
            turns[index].isFinal = isFinal
        } else {
            turns.append(
                ConversationTurn(
                    id: id, speakerLabel: nil, sourceText: text, sourceLocale: locale,
                    translatedText: "", isFinal: isFinal, isTranslationFinal: false, timestamp: timestamp
                )
            )
            applySpeakerLabels()
        }
    }

    private func translateOnDeviceSegment(id: Int, text: String, locale: Locale) {
        guard let sourceLang = SupportedLanguage.byLocaleIdentifier(locale.identifier) else { return }
        let target = settings.targetLanguage
        Task {
            let translated = await translationManager.translate(segmentID: id, text: text, source: sourceLang, target: target)
            if let index = turns.firstIndex(where: { $0.id == id }) {
                turns[index].translatedText = translated
                turns[index].isTranslationFinal = true
            }
        }
    }

    private func applyTranslationDelta(segmentID: Int, delta: String, isFinal: Bool) {
        guard let index = turns.firstIndex(where: { $0.id == segmentID }) else { return }
        turns[index].translatedText += delta
        if isFinal { turns[index].isTranslationFinal = true }
    }

    /// Diarization resolves ~5s analysis windows well after the audio (and
    /// often after the transcript segment) arrives, so this is re-run
    /// every time either a new SpeakerTurn or a new transcript turn shows
    /// up. Point-based matching against speakerTurns[].contains(timestamp),
    /// falling back to the nearest turn within 2s, is an approximation --
    /// exact sample-accurate alignment isn't attempted since diart's own
    /// window granularity (~500ms shift) already bounds the achievable
    /// precision. See ios/README.md for expected behavior.
    private func applySpeakerLabels() {
        for index in turns.indices where turns[index].speakerLabel == nil {
            let t = turns[index].timestamp
            if let match = speakerTurns.first(where: { $0.contains(t) }) {
                turns[index].speakerLabel = match.speaker
            } else if let nearest = speakerTurns.min(by: { abs($0.midpoint - t) < abs($1.midpoint - t) }),
                      abs(nearest.midpoint - t) < 2.0 {
                turns[index].speakerLabel = nearest.speaker
            }
        }
    }
}
