import AVFoundation
import Foundation
import Speech

/// Owns the active SFSpeechRecognizer + recognition task, restarting with a
/// new locale whenever LanguageSwitchDetector signals a language change.
///
/// SFSpeechRecognizer has no built-in auto-detect or code-switching support
/// -- a recognizer is bound to one fixed locale for its whole lifetime -- so
/// this cancel-and-restart-with-a-new-locale approach is the standard
/// workaround for bilingual/multilingual live captioning. It costs a brief
/// gap in transcription right at the switch point, which is an accepted
/// trade-off (see ios/README.md).
final class SpeechRecognitionManager {
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private(set) var currentLocale: Locale
    private var segmentCounter = 0
    private var generation = 0
    private var sessionStartTime: Date?

    private var continuation: AsyncStream<TranscriptSegment>.Continuation?
    lazy var segments: AsyncStream<TranscriptSegment> = AsyncStream { continuation in
        self.continuation = continuation
    }

    init(initialLocale: Locale) {
        currentLocale = initialLocale
    }

    static func requestAuthorization() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }

    /// Whether every candidate locale has an on-device-capable recognizer.
    /// If any is missing, the caller should use the /ws/stream cloud
    /// fallback for the whole session instead of on-device STT.
    static func supportsOnDevice(locales: [Locale]) -> Bool {
        locales.allSatisfy { locale in
            guard let recognizer = SFSpeechRecognizer(locale: locale) else { return false }
            return recognizer.supportsOnDeviceRecognition
        }
    }

    func start(sessionStartTime: Date) {
        self.sessionStartTime = sessionStartTime
        beginRecognition(locale: currentLocale)
    }

    /// Called by LanguageSwitchDetector when the spoken language changes.
    func switchLocale(to locale: Locale) {
        guard locale.identifier(.bcp47) != currentLocale.identifier(.bcp47) else { return }
        currentLocale = locale
        beginRecognition(locale: locale)
    }

    private func beginRecognition(locale: Locale) {
        generation += 1
        let generationAtStart = generation

        task?.cancel()
        request?.endAudio()

        guard let recognizer = SFSpeechRecognizer(locale: locale) else { return }
        recognizer.defaultTaskHint = .dictation
        self.recognizer = recognizer

        let newRequest = SFSpeechAudioBufferRecognitionRequest()
        newRequest.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition {
            newRequest.requiresOnDeviceRecognition = true
        }
        request = newRequest

        task = recognizer.recognitionTask(with: newRequest) { [weak self] result, _ in
            guard let self, self.generation == generationAtStart, let result else { return }
            let text = result.bestTranscription.formattedString
            guard !text.isEmpty else { return }

            let elapsed = self.sessionStartTime.map { Date().timeIntervalSince($0) } ?? 0
            let isFinal = result.isFinal
            let id = isFinal ? self.nextSegmentID() : -1

            self.continuation?.yield(
                TranscriptSegment(id: id, text: text, isFinal: isFinal, locale: locale, timestamp: elapsed)
            )
        }
    }

    private func nextSegmentID() -> Int {
        segmentCounter += 1
        return segmentCounter
    }

    /// Feed a native-format audio buffer captured by AudioCaptureEngine.
    func append(_ buffer: AVAudioPCMBuffer) {
        request?.append(buffer)
    }

    func stop() {
        generation += 1 // invalidate any in-flight result callbacks
        task?.cancel()
        request?.endAudio()
        task = nil
        request = nil
        continuation?.finish()
    }
}
