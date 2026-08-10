import Foundation
import Translation

/// Wraps Apple's on-device Translation framework, falling back to the
/// backend's Groq-based /ws/translate for language pairs it doesn't support.
///
/// TranslationSession can only be obtained from SwiftUI's `.translationTask`
/// view modifier, so this class doesn't create one itself: it publishes the
/// `(source, target)` pair it currently needs via `requestedConfiguration`,
/// and expects a view in the hierarchy (see ConversationView) to host
/// `.translationTask(requestedConfiguration) { session in manager.attach(session) }`
/// and hand the resulting session back via `attach(session:)`.
@MainActor
final class TranslationManager: ObservableObject {
    @Published var requestedConfiguration: TranslationSession.Configuration?

    private var session: TranslationSession?
    private var configuredSource: Locale.Language?
    private var configuredTarget: Locale.Language?

    private let backend: BackendClient
    private var fallbackSocket: BackendSocket?
    private var pendingFallback: [Int: CheckedContinuation<String, Never>] = [:]

    init(backendConfig: BackendConfig) {
        backend = BackendClient(config: backendConfig)
    }

    /// Call whenever the active source/target language pair changes (session
    /// start, manual override, or a language-switch from LanguageSwitchDetector).
    func configure(source: Locale.Language, target: Locale.Language) {
        guard configuredSource != source || configuredTarget != target else { return }
        configuredSource = source
        configuredTarget = target
        requestedConfiguration = TranslationSession.Configuration(source: source, target: target)
    }

    /// Called by the hosting view's `.translationTask` once a session is ready.
    func attach(session: TranslationSession) {
        self.session = session
    }

    func translate(segmentID: Int, text: String, source: SupportedLanguage, target: SupportedLanguage) async -> String {
        let availability = LanguageAvailability()
        let status = await availability.status(from: source.language, to: target.language)

        switch status {
        case .installed:
            if let session, configuredSource == source.language, configuredTarget == target.language,
               let translated = try? await session.translate(text) {
                return translated.targetText
            }
            return await translateViaBackend(segmentID: segmentID, text: text, target: target)
        case .supported:
            // Assets aren't downloaded yet; don't block a live caption on a
            // multi-hundred-MB download mid-conversation -- fall back now,
            // the pair becomes .installed for later segments once the
            // hosting view's .translationTask finishes downloading it.
            return await translateViaBackend(segmentID: segmentID, text: text, target: target)
        case .unsupported:
            return await translateViaBackend(segmentID: segmentID, text: text, target: target)
        @unknown default:
            return await translateViaBackend(segmentID: segmentID, text: text, target: target)
        }
    }

    private func translateViaBackend(segmentID: Int, text: String, target: SupportedLanguage) async -> String {
        if fallbackSocket == nil {
            guard let socket = backend.openTranslateSocket() else { return text }
            fallbackSocket = socket
            socket.connect()
            listenForFallbackResults(on: socket)
        }
        guard let socket = fallbackSocket else { return text }

        return await withCheckedContinuation { continuation in
            pendingFallback[segmentID] = continuation
            Task {
                try? await socket.sendJSON(["segment_id": segmentID, "text": text, "target_lang": target.displayName])
            }
        }
    }

    private func listenForFallbackResults(on socket: BackendSocket) {
        Task {
            var accumulated: [Int: String] = [:]
            do {
                for try await message in socket.messages {
                    guard message["type"] as? String == "translation",
                          let segmentID = message["segment_id"] as? Int else { continue }
                    let delta = message["data"] as? String ?? ""
                    let isFinal = message["final"] as? Bool ?? false
                    accumulated[segmentID, default: ""] += delta
                    if isFinal, let continuation = pendingFallback.removeValue(forKey: segmentID) {
                        continuation.resume(returning: accumulated.removeValue(forKey: segmentID) ?? "")
                    }
                }
            } catch {
                // Connection dropped; resolve any pending requests so callers
                // don't hang forever on a continuation that'll never fire.
            }
            for continuation in pendingFallback.values {
                continuation.resume(returning: "")
            }
            pendingFallback.removeAll()
        }
    }

    func stop() {
        fallbackSocket?.close()
        fallbackSocket = nil
        for continuation in pendingFallback.values {
            continuation.resume(returning: "")
        }
        pendingFallback.removeAll()
    }
}
