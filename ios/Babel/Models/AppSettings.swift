import Foundation

/// User-configurable language settings (see SettingsView), persisted across
/// launches. Distinct from BackendConfig (connection settings).
final class AppSettings: ObservableObject {
    private static let candidatesKey = "babel.candidateLanguages"
    private static let targetKey = "babel.targetLanguage"

    /// The set of languages the speaker(s) might use. On-device STT only
    /// switches between these (see LanguageSwitchDetector); the cloud
    /// fallback path (Deepgram has no auto-detect either) always uses just
    /// the first one for the whole session.
    @Published var candidateLocaleIdentifiers: [String] {
        didSet { UserDefaults.standard.set(candidateLocaleIdentifiers, forKey: Self.candidatesKey) }
    }
    @Published var targetLocaleIdentifier: String {
        didSet { UserDefaults.standard.set(targetLocaleIdentifier, forKey: Self.targetKey) }
    }

    init() {
        candidateLocaleIdentifiers = UserDefaults.standard.stringArray(forKey: Self.candidatesKey)
            ?? [SupportedLanguage.all[0].localeIdentifier] // English
        targetLocaleIdentifier = UserDefaults.standard.string(forKey: Self.targetKey)
            ?? SupportedLanguage.defaultTarget.localeIdentifier
    }

    var candidateLanguages: [SupportedLanguage] {
        let matched = candidateLocaleIdentifiers.compactMap(SupportedLanguage.byLocaleIdentifier)
        return matched.isEmpty ? [SupportedLanguage.all[0]] : matched
    }

    var targetLanguage: SupportedLanguage {
        SupportedLanguage.byLocaleIdentifier(targetLocaleIdentifier) ?? .defaultTarget
    }

    func toggleCandidate(_ language: SupportedLanguage) {
        if let index = candidateLocaleIdentifiers.firstIndex(of: language.localeIdentifier) {
            guard candidateLocaleIdentifiers.count > 1 else { return } // always keep at least one
            candidateLocaleIdentifiers.remove(at: index)
        } else {
            candidateLocaleIdentifiers.append(language.localeIdentifier)
        }
    }
}
