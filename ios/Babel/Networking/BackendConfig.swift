import Foundation

/// User-configurable backend connection settings (see SettingsView),
/// persisted across launches. Defaults to the FastAPI dev server address
/// used throughout backend/README.md.
final class BackendConfig: ObservableObject {
    private static let storageKey = "babel.backendBaseURL"

    @Published var baseURLString: String {
        didSet { UserDefaults.standard.set(baseURLString, forKey: Self.storageKey) }
    }

    init() {
        baseURLString = UserDefaults.standard.string(forKey: Self.storageKey) ?? "ws://localhost:8000"
    }

    func wsURL(path: String, query: [String: String] = [:]) -> URL? {
        var components = URLComponents(string: baseURLString + path)
        if !query.isEmpty {
            components?.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }
        }
        return components?.url
    }
}
