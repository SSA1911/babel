import Foundation

/// A language the app knows how to work with, carrying every representation
/// the different frameworks/backends need:
///  - `deepgramCode`: short code matching backend `SUPPORTED_SOURCE_LANGUAGES`
///    (app/main.py), used only for the /ws/stream cloud-STT-fallback path.
///  - `localeIdentifier`: full BCP-47 identifier for `SFSpeechRecognizer`
///    and the `Translation` framework, which need a region to pick a model.
///  - `displayName`: human-readable name sent verbatim to /ws/translate and
///    /ws/stream, since translator.py's Groq prompt expects a name like
///    "Spanish", not a code (mirrors frontend/index.html's dropdown values).
struct SupportedLanguage: Identifiable, Hashable {
    let deepgramCode: String
    let localeIdentifier: String
    let displayName: String

    var id: String { localeIdentifier }
    var locale: Locale { Locale(identifier: localeIdentifier) }
    var language: Locale.Language { Locale.Language(identifier: localeIdentifier) }

    static let all: [SupportedLanguage] = [
        SupportedLanguage(deepgramCode: "en", localeIdentifier: "en-US", displayName: "English"),
        SupportedLanguage(deepgramCode: "es", localeIdentifier: "es-ES", displayName: "Spanish"),
        SupportedLanguage(deepgramCode: "fr", localeIdentifier: "fr-FR", displayName: "French"),
        SupportedLanguage(deepgramCode: "de", localeIdentifier: "de-DE", displayName: "German"),
        SupportedLanguage(deepgramCode: "zh", localeIdentifier: "zh-Hans", displayName: "Mandarin Chinese"),
        SupportedLanguage(deepgramCode: "ja", localeIdentifier: "ja-JP", displayName: "Japanese"),
        SupportedLanguage(deepgramCode: "ko", localeIdentifier: "ko-KR", displayName: "Korean"),
        SupportedLanguage(deepgramCode: "pt", localeIdentifier: "pt-PT", displayName: "Portuguese"),
        SupportedLanguage(deepgramCode: "hi", localeIdentifier: "hi-IN", displayName: "Hindi"),
        SupportedLanguage(deepgramCode: "ar", localeIdentifier: "ar-SA", displayName: "Arabic"),
        SupportedLanguage(deepgramCode: "tr", localeIdentifier: "tr-TR", displayName: "Turkish"),
        SupportedLanguage(deepgramCode: "tl", localeIdentifier: "fil-PH", displayName: "Tagalog"),
        SupportedLanguage(deepgramCode: "ta", localeIdentifier: "ta-IN", displayName: "Tamil"),
    ]

    static let defaultTarget = all[0] // English, matches backend's TARGET_LANGUAGE default

    static func byLocaleIdentifier(_ identifier: String) -> SupportedLanguage? {
        all.first { $0.localeIdentifier == identifier }
    }
}
