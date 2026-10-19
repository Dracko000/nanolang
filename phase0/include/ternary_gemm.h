#ifndef TERNARY_GEMM_H
#define TERNARY_GEMM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Baseline FP32 Matrix-Vector Multiplication: Y = A * X
 * A: M x K (row-major)
 * X: K x 1
 * Y: M x 1
 */
void fp32_gemv_baseline(float *Y, const float *A, const float *X, int M, int K);

/**
 * Baseline INT8 Matrix-Vector Multiplication (with standard multiplication): Y = W * X
 * W: M x K (values in [-128, 127])
 * X: K x 1 (values in [-128, 127])
 * Y: M x 1 (32-bit integer accumulation)
 */
void int8_gemv_baseline(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K);

/**
 * Ternary Reference Kernel: Y = W * X
 * W contains values strictly in {-1, 0, +1}.
 * Implements pure ADD / SUB with ZERO-SKIP. No multiplication operations.
 */
void ternary_gemv_ref(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K);

/**
 * Ternary Bitmask Kernel:
 * Weights are encoded as two separate bitmasks:
 * - mask_pos: bit is 1 if W[i, j] == +1
 * - mask_neg: bit is 1 if W[i, j] == -1
 * Memory required: 2 bits per weight (16x smaller than FP32).
 */
void ternary_gemv_bitmask(int32_t *Y,
                          const uint64_t *mask_pos,
                          const uint64_t *mask_neg,
                          const int8_t *X,
                          int M, int K);

/**
 * Optimized AVX2 SIMD Ternary Kernel (x86_64).
 * Vectorized addition and subtraction using 256-bit AVX2 registers.
 */
void ternary_gemv_avx2(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K);

/**
 * ARM NEON SIMD Ternary Kernel (ARM64 / Apple Silicon / Snapdragon).
 * Defined and compiled on aarch64 targets.
 */
#if defined(__aarch64__) || defined(_M_ARM64)
void ternary_gemv_neon(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K);
#endif

/**
 * End-to-end ternary GEMV with float activations.
 *
 * 1. Quantizes the float activation vector X to int8 (BitNet activation_quant:
 *    per-row scale s = 127 / max|X|).
 * 2. Runs the zero-multiply add/sub ternary kernel over W in {-1, 0, +1}.
 * 3. Dequantizes: Y[i] = (W · Xq)[i] * (scale / s).
 *
 * Y is float32 (M x 1), W is int8 row-major (M x K), X is float32 (K x 1).
 */
void ternary_gemv_f32(float *Y, const int8_t *W, const float *X, int M, int K, float scale);

void ternary_gemv_f32_ref(float *Y, const int8_t *W, const float *X, int M, int K, float scale);

/**
 * End-to-end ternary GEMV over 2-bit packed weights (the .nanow on-disk layout).
 *
 * Wp holds ceil(K/4) bytes per row; each byte packs 4 consecutive weights as
 * 2-bit codes (0 -> 0, 1 -> +1, 3 -> -1) matching the Python packer. Requires
 * K % 4 == 0. Quantization and dequantization match ternary_gemv_f32.
 *
 * It reads 4x fewer weight bytes than the int8 kernel, so it is the
 * bandwidth-optimized path for the packed .nanow container.
 */
void ternary_gemv_packed_f32(float *Y, const uint8_t *Wp, const float *X, int M, int K, float scale);

void ternary_gemv_packed_ref(float *Y, const uint8_t *Wp, const float *X, int M, int K, float scale);

/**
 * Parallel AVX2/FMA float32 GEMV: Y = W @ X with W row-major (M x K). Rows are
 * split across OpenMP threads. Used for the FP32 LM head so the hot path never
 * mixes libgomp and OpenBLAS thread pools (which contend badly on small VMs).
 */
void fp32_gemv_par(float *Y, const float *W, const float *X, int M, int K);

/**
 * Bit-packing helper: converts an array of ternary int8 values {-1, 0, 1}
 * into separate pos and neg bitmasks (64 weights per uint64_t pair).
 */
void pack_ternary_bitmasks(uint64_t *mask_pos,
                           uint64_t *mask_neg,
                           const int8_t *W,
                           int M, int K);

/**
 * Elementwise transformer ops (scalar C; see nano_elementwise.c). These mirror
 * the NumPy reference in run_smollm2.py and exist to remove per-call NumPy
 * overhead from the single-token decode path.
 */
void nano_rmsnorm_f32(float *out, const float *x, const float *w, int n, float eps);
void nano_silu_mul_f32(float *out, const float *gate, const float *up, int n);
void nano_rope_f32(float *x, const float *cos, const float *sin,
                   int n_heads, int head_dim);
void nano_softmax_axis1_f32(float *x, int axis0, int n, int axis2);

#ifdef __cplusplus
}
#endif

#endif // TERNARY_GEMM_H
