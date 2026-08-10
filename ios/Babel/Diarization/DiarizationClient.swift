import Foundation

/// Persistent connection to /ws/diarize: streams the session's resampled
/// PCM16 audio to the backend and exposes resolved SpeakerTurns as they
/// arrive. Always active while a session is running, independent of
/// whether STT is on-device or using the /ws/stream fallback -- Apple has
/// no on-device diarization API, so this is the only source of speaker
/// separation regardless of STT path.
final class DiarizationClient {
    private var socket: BackendSocket?
    private let client: BackendClient

    private var continuation: AsyncStream<SpeakerTurn>.Continuation?
    lazy var turns: AsyncStream<SpeakerTurn> = AsyncStream { continuation in
        self.continuation = continuation
    }

    /// Surfaced to the UI so a HUGGINGFACE_TOKEN/diart-not-installed error
    /// on the backend (see backend/app/diarization.py) is visible rather
    /// than silently leaving every turn unlabeled.
    var onError: ((String) -> Void)?

    init(client: BackendClient) {
        self.client = client
    }

    func start() {
        guard let socket = client.openDiarizationSocket() else {
            onError?("Could not build /ws/diarize URL; check the backend address in Settings.")
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
                onError?("Diarization stream error: \(error.localizedDescription)")
            }
            continuation?.finish()
        }
    }

    private func handle(_ message: [String: Any]) {
        guard let type = message["type"] as? String else { return }
        switch type {
        case "speaker_turn":
            guard let speaker = message["speaker"] as? String,
                  let start = message["start"] as? Double,
                  let end = message["end"] as? Double else { return }
            continuation?.yield(SpeakerTurn(speaker: speaker, start: start, end: end))
        case "error":
            onError?(message["data"] as? String ?? "Unknown diarization error")
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
