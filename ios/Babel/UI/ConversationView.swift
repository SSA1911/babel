import SwiftUI
import Translation

struct ConversationView: View {
    @StateObject private var settings: AppSettings
    @StateObject private var backendConfig: BackendConfig
    @StateObject private var session: ConversationSession
    @State private var showingSettings = false

    init() {
        let settings = AppSettings()
        let backendConfig = BackendConfig()
        _settings = StateObject(wrappedValue: settings)
        _backendConfig = StateObject(wrappedValue: backendConfig)
        _session = StateObject(wrappedValue: ConversationSession(settings: settings, backendConfig: backendConfig))
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                statusBar
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(session.turns) { turn in
                                TurnBubble(turn: turn).id(turn.id)
                            }
                            if !session.interimText.isEmpty {
                                InterimBubble(text: session.interimText, locale: session.interimLocale)
                                    .id(Int.min)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: session.turns.count) { _, _ in
                        withAnimation {
                            proxy.scrollTo(session.turns.last?.id ?? Int.min, anchor: .bottom)
                        }
                    }
                }
                controls
            }
            .navigationTitle("Babel")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                    .disabled(session.isRunning)
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView(settings: settings, backendConfig: backendConfig)
            }
            // Hosts the on-device Translation session TranslationManager
            // needs; see TranslationManager's doc comment for why this has
            // to live in the view layer rather than the manager itself.
            .translationTask(session.translationManager.requestedConfiguration) { translationSession in
                session.translationManager.attach(session: translationSession)
                try? await translationSession.prepareTranslation()
            }
        }
    }

    private var statusBar: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(session.isRunning ? Color.green : Color.gray)
                .frame(width: 8, height: 8)
            Text(session.statusMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            if let error = session.lastError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .multilineTextAlignment(.trailing)
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var controls: some View {
        Button {
            Task {
                if session.isRunning {
                    session.stop()
                } else {
                    await session.start()
                }
            }
        } label: {
            Label(session.isRunning ? "Stop" : "Start Listening", systemImage: session.isRunning ? "stop.fill" : "mic.fill")
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .tint(session.isRunning ? .red : .accentColor)
        .padding()
    }
}

private struct TurnBubble: View {
    let turn: ConversationTurn

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                if let speaker = turn.speakerLabel {
                    Text(speaker)
                        .font(.caption2.bold())
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(color(for: speaker).opacity(0.18))
                        .foregroundStyle(color(for: speaker))
                        .clipShape(Capsule())
                }
                Text(languageTag(turn.sourceLocale))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            Text(turn.sourceText)
                .font(.body)
            if !turn.translatedText.isEmpty {
                Text(turn.translatedText)
                    .font(.body.italic())
                    .foregroundStyle(.blue)
            } else if turn.isFinal {
                ProgressView()
                    .scaleEffect(0.6)
                    .frame(height: 12, alignment: .leading)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func languageTag(_ locale: Locale) -> String {
        locale.language.languageCode?.identifier.uppercased() ?? locale.identifier.uppercased()
    }

    private func color(for speaker: String) -> Color {
        let hash = abs(speaker.hashValue)
        let hue = Double(hash % 360) / 360.0
        return Color(hue: hue, saturation: 0.55, brightness: 0.8)
    }
}

private struct InterimBubble: View {
    let text: String
    let locale: Locale?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let locale {
                Text(locale.language.languageCode?.identifier.uppercased() ?? "")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(text)
                .font(.body.italic())
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
    }
}
