import Foundation

/// One interim/final result from SpeechRecognitionManager. `timestamp` is
/// elapsed seconds since the session's shared audio clock started (see
/// AudioCaptureEngine.sessionStartTime), used to align against SpeakerTurns.
struct TranscriptSegment {
    let id: Int
    let text: String
    let isFinal: Bool
    let locale: Locale
    let timestamp: TimeInterval
}

/// One resolved speaker turn from DiarizationClient, in the same
/// session-relative time base as TranscriptSegment.timestamp.
struct SpeakerTurn {
    let speaker: String
    let start: TimeInterval
    let end: TimeInterval

    func contains(_ time: TimeInterval) -> Bool {
        time >= start && time <= end
    }

    var midpoint: TimeInterval { (start + end) / 2 }
}
