import Foundation

enum FallbackStreamEvent {
    case transcript(TranscriptSegment)
    case translationDelta(segmentID: Int, text: String, isFinal: Bool)
    case error(String)
}

/// Wraps the existing /ws/stream (Deepgram STT + Groq translation) for
/// whole-session use when a configured source language has no on-device
/// SFSpeechRecognizer. Same 16kHz mono PCM16 audio format as
/// DiarizationClient, so AudioCaptureEngine's resampled output feeds both.
final class FallbackStreamClient {
    private var socket: BackendSocket?
    private let client: BackendClient
    private let source: SupportedLanguage
    private let target: SupportedLanguage

    private var continuation: AsyncStream<FallbackStreamEvent>.Continuation?
    lazy var events: AsyncStream<FallbackStreamEvent> = AsyncStream { continuation in
        self.continuation = continuation
    }

    init(client: BackendClient, source: SupportedLanguage, target: SupportedLanguage) {
        self.client = client
        self.source = source
        self.target = target
    }

    func start() {
        guard let socket = client.openFallbackStreamSocket(source: source, target: target) else {
            continuation?.yield(.error("Could not build /ws/stream URL; check the backend address in Settings."))
            continuation?.finish()
            return
        }
        self.socket = socket
        socket.connect()

        Task {
            do {
                for try await message in socket.messages {
                    handle(message)
                }
            } catch {
                continuation?.yield(.error("Fallback stream error: \(error.localizedDescription)"))
            }
            continuation?.finish()
        }
    }

    private func handle(_ message: [String: Any]) {
        guard let type = message["type"] as? String else { return }
        switch type {
        case "transcript":
            guard let text = message["data"] as? String,
                  let isFinal = message["is_final"] as? Bool,
                  let segmentID = message["segment_id"] as? Int else { return }
            // Deepgram doesn't hand back a session-relative timestamp;
            // ConversationSession fills one in (using its shared audio
            // clock) when it receives this segment, same as it does for
            // on-device results.
            let segment = TranscriptSegment(id: segmentID, text: text, isFinal: isFinal, locale: source.locale, timestamp: 0)
            continuation?.yield(.transcript(segment))
        case "translation":
            guard let segmentID = message["segment_id"] as? Int else { return }
            let text = message["data"] as? String ?? ""
            let final = message["final"] as? Bool ?? false
            continuation?.yield(.translationDelta(segmentID: segmentID, text: text, isFinal: final))
        case "error":
            continuation?.yield(.error(message["data"] as? String ?? "Unknown fallback stream error"))
        default:
            break
        }
    }

    func push(_ chunk: Data) {
        Task { try? await socket?.send(data: chunk) }
    }

    func stop() {
        socket?.close()
        continuation?.finish()
    }
}
