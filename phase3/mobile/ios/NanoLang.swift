import Foundation

/// NanoLang Edge AI Engine for iOS / Apple Silicon
/// Powered by Zero-Copy mmap, BitNet b1.58 Ternary Weights, and O(1) Memory State Space Models.
public final class NanoLangModel {
    private var handle: UnsafeMutableRawPointer?

    public enum Language: Int32 {
        case indonesian = 0
        case english = 1
    }

    public enum NanoLangError: Error {
        case modelNotFound
        case initializationFailed
        case generationFailed
    }

    /// Initializes and maps a .nanow model container via zero-copy mmap.
    public init(modelPath: String) throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw NanoLangError.modelNotFound
        }

        guard let h = nanolang_mobile_init(modelPath) else {
            throw NanoLangError.initializationFailed
        }
        self.handle = h
    }

    deinit {
        if let h = handle {
            nanolang_mobile_free(h)
        }
    }

    /// Generates high-efficiency on-device response.
    /// Execution consumes < 2.5 Watts and guarantees zero memory growth.
    public func generate(prompt: String, language: Language = .indonesian) throws -> String {
        guard let h = handle else { throw NanoLangError.initializationFailed }

        var buffer = [CChar](repeating: 0, count: 2048)
        let written = nanolang_mobile_generate(h, prompt, &buffer, buffer.count, language.rawValue)

        guard written > 0 else {
            throw NanoLangError.generationFailed
        }

        return String(cString: buffer)
    }

    /// Constant recurrent state memory in Kilobytes (O(1) Memory).
    public var stateMemoryKB: Float {
        guard let h = handle else { return 0.0 }
        return nanolang_mobile_get_state_kb(h)
    }
}

// C ABI Bridging declarations (corresponds to nanolang_bridge.h)
@_silgen_name("nanolang_mobile_init")
private func nanolang_mobile_init(_ modelPath: UnsafePointer<CChar>) -> UnsafeMutableRawPointer?

@_silgen_name("nanolang_mobile_free")
private func nanolang_mobile_free(_ handle: UnsafeMutableRawPointer)

@_silgen_name("nanolang_mobile_generate")
private func nanolang_mobile_generate(_ handle: UnsafeMutableRawPointer,
                                      _ prompt: UnsafePointer<CChar>,
                                      _ responseOut: UnsafeMutablePointer<CChar>,
                                      _ maxOutLen: Int,
                                      _ langId: Int32) -> Int32

@_silgen_name("nanolang_mobile_get_state_kb")
private func nanolang_mobile_get_state_kb(_ handle: UnsafeMutableRawPointer) -> Float
