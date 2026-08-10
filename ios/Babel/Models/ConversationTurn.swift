import Foundation

/// One row of the live conversation UI: a transcribed (and, once ready,
/// translated) segment, tagged with a speaker label once diarization
/// resolves the analysis window it falls in.
struct ConversationTurn: Identifiable, Equatable {
    let id: Int
    var speakerLabel: String?
    var sourceText: String
    var sourceLocale: Locale
    var translatedText: String
    var isFinal: Bool
    var isTranslationFinal: Bool
    let timestamp: TimeInterval
}
