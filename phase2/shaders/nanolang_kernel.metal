#include <metal_stdlib>
using namespace metal;

/**
 * Apple Metal Compute Shader: Zero-Multiplication 2-bit Ternary GEMV
 * Target: Apple Silicon Unified Memory Architecture (UMA)
 *
 * Each threadgroup processes a tile of output rows.
 * Weights are 2-bit packed: 4 weights per byte.
 * Arithmetic: Pure Integer Addition & Subtraction (Zero Floating-Point MAC).
 */
kernel void ternary_gemv_2bit(
    device int32_t*               Y           [[buffer(0)]], // Output vector [M]
    device const uint8_t*         packed_W    [[buffer(1)]], // Packed 2-bit weights [M x (K / 4)]
    device const int8_t*          X           [[buffer(2)]], // Input INT8 activations [K]
    constant uint32_t&            M           [[buffer(3)]],
    constant uint32_t&            K           [[buffer(4)]],
    uint2                         thread_pos  [[thread_position_in_grid]],
    uint2                         tg_pos      [[thread_position_in_threadgroup]],
    uint2                         tg_size     [[threads_per_threadgroup]])
{
    uint row = thread_pos.x;
    if (row >= M) return;

    int32_t accumulator = 0;
    uint bytes_per_row = (K + 3) / 4;
    device const uint8_t* row_weights = packed_W + row * bytes_per_row;

    // Process 4 weights per packed byte
    for (uint b = 0; b < bytes_per_row; ++b) {
        uint8_t packed_byte = row_weights[b];
        uint k_base = b * 4;

        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            uint k = k_base + i;
            if (k < K) {
                uint8_t code = (packed_byte >> (i * 2)) & 0x03;
                int8_t act = X[k];

                // 00b -> 0 (Skip)
                // 01b -> +1 (Add)
                // 11b -> -1 (Subtract)
                if (code == 1) {
                    accumulator += (int32_t)act;
                } else if (code == 3) {
                    accumulator -= (int32_t)act;
                }
            }
        }
    }

    Y[row] = accumulator;
}
