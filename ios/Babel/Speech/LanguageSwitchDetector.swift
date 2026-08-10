import Foundation
import NaturalLanguage

/// Detects when the spoken language has likely switched, by running
/// NLLanguageRecognizer over stabilized interim transcript text and
/// comparing against the currently active recognition locale.
///
/// This exists because SFSpeechRecognizer cannot auto-detect or code-switch
/// languages on its own -- it only ever transcribes in the one locale it
/// was created with. Biasing detection to the user's configured candidate
/// languages (Settings), requiring a confidence margin, and requiring the
/// same candidate language to win for `stableThreshold` consecutive interim
/// updates all exist to avoid flapping between locales on short or
/// ambiguous fragments.
final class LanguageSwitchDetector {
    private let localeByLanguage: [NLLanguage: Locale]
    private let minConfidenceMargin: Double
    private let stableThreshold: Int

    private var lastCandidateLanguage: NLLanguage?
    private var stableCount = 0

    init(candidateLocales: [Locale], minConfidenceMargin: Double = 0.15, stableThreshold: Int = 2) {
        var map: [NLLanguage: Locale] = [:]
        for locale in candidateLocales {
            let code = locale.language.languageCode?.identifier ?? locale.identifier
            map[NLLanguage(code)] = locale
        }
        localeByLanguage = map
        self.minConfidenceMargin = minConfidenceMargin
        self.stableThreshold = stableThreshold
    }

    /// Feed each interim/final transcript update. Returns a Locale to
    /// switch to, or nil if no switch is warranted (yet).
    func evaluate(text: String, currentLocale: Locale) -> Locale? {
        // Only one candidate language configured: nothing to switch to.
        // Very short fragments are unreliable for language ID; wait for more text.
        guard localeByLanguage.count > 1, text.count >= 8 else { return nil }

        let candidateLanguages = Array(localeByLanguage.keys)
        let recognizer = NLLanguageRecognizer()
        recognizer.languageHints = Dictionary(
            uniqueKeysWithValues: candidateLanguages.map { ($0, 1.0 / Double(candidateLanguages.count)) }
        )
        recognizer.languageConstraints = candidateLanguages
        recognizer.processString(text)

        let hypotheses = recognizer.languageHypotheses(withMaximum: candidateLanguages.count)
        guard let top = hypotheses.max(by: { $0.value < $1.value })?.key else { return nil }

        let currentCode = currentLocale.language.languageCode?.identifier ?? currentLocale.identifier
        let currentLanguage = NLLanguage(currentCode)

        guard top != currentLanguage else {
            resetStability()
            return nil
        }

        let topConfidence = hypotheses[top] ?? 0
        let currentConfidence = hypotheses[currentLanguage] ?? 0
        guard topConfidence - currentConfidence >= minConfidenceMargin else {
            resetStability()
            return nil
        }

        if lastCandidateLanguage == top {
            stableCount += 1
        } else {
            lastCandidateLanguage = top
            stableCount = 1
        }

        guard stableCount >= stableThreshold, let newLocale = localeByLanguage[top] else { return nil }

        resetStability()
        return newLocale
    }

    private func resetStability() {
        stableCount = 0
        lastCandidateLanguage = nil
    }
}
