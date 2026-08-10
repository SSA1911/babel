import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: AppSettings
    @ObservedObject var backendConfig: BackendConfig
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("ws://host:8000", text: $backendConfig.baseURLString)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                } header: {
                    Text("Backend")
                } footer: {
                    Text("Used for speaker diarization always, and as a fallback for STT/translation when Apple's on-device tools can't handle a language pair.")
                }

                Section("Target language") {
                    Picker("Translate into", selection: $settings.targetLocaleIdentifier) {
                        ForEach(SupportedLanguage.all) { language in
                            Text(language.displayName).tag(language.localeIdentifier)
                        }
                    }
                }

                Section {
                    ForEach(SupportedLanguage.all) { language in
                        Toggle(
                            language.displayName,
                            isOn: Binding(
                                get: { settings.candidateLocaleIdentifiers.contains(language.localeIdentifier) },
                                set: { _ in settings.toggleCandidate(language) }
                            )
                        )
                    }
                } header: {
                    Text("Spoken languages")
                } footer: {
                    Text("Select every language you expect to hear. With two or more selected and an on-device recognizer available for all of them, Babel switches automatically as speakers change languages mid-conversation. The cloud fallback path has no auto-detect, so it always uses just the first selected language for the whole session.")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
