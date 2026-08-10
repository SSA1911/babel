import AVFoundation
import Foundation

/// Captures microphone audio via AVAudioEngine and fans it out to:
///  (a) on-device speech recognition, as native-format buffers (Apple's own
///      sample code appends hardware-format buffers directly to
///      SFSpeechAudioBufferRecognitionRequest -- no resampling needed there), and
///  (b) network consumers, as resampled 16kHz mono PCM16 (the format both
///      /ws/diarize and /ws/stream expect).
///
/// `sessionStartTime` is the shared clock every other component timestamps
/// against (TranscriptSegment.timestamp, SpeakerTurn start/end via the
/// backend's own elapsed-seconds-since-connect clock), so diarization and
/// transcript results can be aligned later in ConversationSession.
final class AudioCaptureEngine {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?

    private(set) var sessionStartTime: Date?

    var onNativeBuffer: ((AVAudioPCMBuffer) -> Void)?
    var onPCM16Chunk: ((Data, TimeInterval) -> Void)?

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .allowBluetooth])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        let targetFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)
        self.targetFormat = targetFormat
        if let targetFormat {
            converter = AVAudioConverter(from: inputFormat, to: targetFormat)
        }

        sessionStartTime = Date()

        inputNode.installTap(onBus: 0, bufferSize: 2048, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            self.onNativeBuffer?(buffer)
            self.emitPCM16(from: buffer, inputFormat: inputFormat)
        }

        engine.prepare()
        try engine.start()
    }

    private func emitPCM16(from buffer: AVAudioPCMBuffer, inputFormat: AVAudioFormat) {
        guard let converter, let targetFormat, let sessionStartTime else { return }

        let ratio = targetFormat.sampleRate / inputFormat.sampleRate
        let outputCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputCapacity) else { return }

        var consumed = false
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }

        guard status == .haveData, conversionError == nil, let channelData = outputBuffer.int16ChannelData else {
            return // drop this chunk rather than interrupt a live session over a transient conversion hiccup
        }

        let frameCount = Int(outputBuffer.frameLength)
        let data = Data(bytes: channelData[0], count: frameCount * MemoryLayout<Int16>.size)
        let elapsed = Date().timeIntervalSince(sessionStartTime)
        onPCM16Chunk?(data, elapsed)
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        sessionStartTime = nil
    }
}
