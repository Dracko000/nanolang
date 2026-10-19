package ai.nanolang

/**
 * NanoLang Edge AI Engine for Android / Qualcomm Snapdragon & MediaTek
 * Zero-Copy mmap, BitNet b1.58 Ternary Weights, and Invariant O(1) Memory Footprint.
 */
class NanoLangModel private constructor(private var handle: Long) : AutoCloseable {

    enum class Language(val id: Int) {
        INDONESIAN(0),
        ENGLISH(1)
    }

    companion object {
        init {
            System.loadLibrary("nanolang")
        }

        @JvmStatic
        fun load(modelPath: String): NanoLangModel {
            val h = nativeInit(modelPath)
            if (h == 0L) {
                throw IllegalStateException("Failed to load NanoLang model from: $modelPath")
            }
            return NanoLangModel(h)
        }

        @JvmStatic
        private external fun nativeInit(modelPath: String): Long
        @JvmStatic
        private external fun nativeFree(handle: Long)
        @JvmStatic
        private external fun nativeGenerate(handle: Long, prompt: String, langId: Int): String
        @JvmStatic
        private external fun nativeGetStateKb(handle: Long): Float
    }

    /**
     * Generates on-device response with < 2.5W energy consumption and 0 KV-cache growth.
     */
    fun generate(prompt: String, language: Language = Language.INDONESIAN): String {
        check(handle != 0L) { "Model has been closed." }
        return nativeGenerate(handle, prompt, language.id)
    }

    /**
     * Invariant recurrent memory size in KB (Never grows during chat).
     */
    val stateMemoryKB: Float
        get() {
            check(handle != 0L) { "Model has been closed." }
            return nativeGetStateKb(handle)
        }

    override fun close() {
        if (handle != 0L) {
            nativeFree(handle)
            handle = 0L
        }
    }
}
