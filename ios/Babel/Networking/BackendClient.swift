import Foundation

/// Thin wrapper around URLSessionWebSocketTask exposing send/receive as
/// async APIs, used by DiarizationClient, TranslationManager's backend
/// fallback, and FallbackStreamClient. Messages are decoded as loosely
/// typed JSON dictionaries (matching the backend's dynamic
/// `{"type": ..., ...}` message shapes in app/schemas.py) rather than
/// Codable structs, so a field the backend adds later doesn't require a
/// matching client change to keep parsing successfully.
final class BackendSocket {
    private let task: URLSessionWebSocketTask
    private var isOpen = false
    private var continuation: AsyncThrowingStream<[String: Any], Error>.Continuation?

    /// Single async sequence of decoded JSON message dictionaries, created
    /// once and fed by the pump task connect() starts -- deliberately not a
    /// computed property, so accessing it more than once (as every caller
    /// here does implicitly via `for try await ... in socket.messages`)
    /// can't accidentally spin up a second pump racing the first one for
    /// task.receive() calls.
    private(set) lazy var messages: AsyncThrowingStream<[String: Any], Error> = AsyncThrowingStream { continuation in
        self.continuation = continuation
    }

    init(url: URL, session: URLSession = .shared) {
        task = session.webSocketTask(with: url)
    }

    func connect() {
        task.resume()
        isOpen = true
        pump()
    }

    private func pump() {
        Task {
            while isOpen {
                do {
                    let message = try await task.receive()
                    if let object = Self.decode(message) {
                        continuation?.yield(object)
                    }
                } catch {
                    isOpen = false
                    continuation?.finish(throwing: error)
                    return
                }
            }
            continuation?.finish()
        }
    }

    func send(data: Data) async throws {
        try await task.send(.data(data))
    }

    func sendJSON(_ object: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        try await task.send(.data(data))
    }

    private static func decode(_ message: URLSessionWebSocketTask.Message) -> [String: Any]? {
        let data: Data
        switch message {
        case .data(let d): data = d
        case .string(let s): data = Data(s.utf8)
        @unknown default: return nil
        }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    func close() {
        isOpen = false
        task.cancel(with: .normalClosure, reason: nil)
    }
}

/// Opens the backend WebSocket connections the app can fall back to.
/// Building the connection lives here; interpreting each endpoint's
/// message shape lives in the type that owns that conversation
/// (DiarizationClient, TranslationManager, FallbackStreamClient).
struct BackendClient {
    let config: BackendConfig

    func openDiarizationSocket() -> BackendSocket? {
        guard let url = config.wsURL(path: "/ws/diarize") else { return nil }
        return BackendSocket(url: url)
    }

    func openTranslateSocket() -> BackendSocket? {
        guard let url = config.wsURL(path: "/ws/translate") else { return nil }
        return BackendSocket(url: url)
    }

    func openFallbackStreamSocket(source: SupportedLanguage, target: SupportedLanguage) -> BackendSocket? {
        guard let url = config.wsURL(
            path: "/ws/stream",
            query: ["lang": target.displayName, "source_lang": source.deepgramCode]
        ) else { return nil }
        return BackendSocket(url: url)
    }
}
