#include "ternary_gemm.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void ternary_gemv_avx2(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K) {
    const __m256i ones_u8 = _mm256_set1_epi8(1);
    const __m256i ones_i16 = _mm256_set1_epi16(1);

    for (int i = 0; i < M; ++i) {
        const int8_t *row = W + i * K;
        __m256i acc32 = _mm256_setzero_si256();
        int j = 0;

        // Process 32 elements per iteration
        for (; j <= K - 32; j += 32) {
            __m256i vx = _mm256_loadu_si256((const __m256i *)(X + j));
            __m256i vw = _mm256_loadu_si256((const __m256i *)(row + j));

            // _mm256_sign_epi8(a, b):
            // If b > 0 -> +a, If b < 0 -> -a, If b == 0 -> 0.
            // Executes ternary selection/negation in 1 cycle without multiplier hardware.
            __m256i signed_act = _mm256_sign_epi8(vx, vw);

            // Horizontal add pairs of int8 into int16
            __m256i pairs16 = _mm256_maddubs_epi16(ones_u8, signed_act);

            // Horizontal add pairs of int16 into int32 and accumulate
            __m256i pairs32 = _mm256_madd_epi16(pairs16, ones_i16);
            acc32 = _mm256_add_epi32(acc32, pairs32);
        }

        // Horizontal reduction of acc32 (8 x 32-bit integers)
        __m128i lo = _mm256_castsi256_si128(acc32);
        __m128i hi = _mm256_extracti128_si256(acc32, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int32_t total = _mm_cvtsi128_si32(sum128);

        // Process tail elements scalar
        for (; j < K; ++j) {
            int8_t w = row[j];
            if (w == 1) total += (int32_t)X[j];
            else if (w == -1) total -= (int32_t)X[j];
        }

        Y[i] = total;
    }
}

void ternary_gemv_f32(float *Y, const int8_t *W, const float *X, int M, int K, float scale) {
    float amax = 0.0f;
    for (int j = 0; j < K; ++j) {
        float a = fabsf(X[j]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-8f) {
        for (int i = 0; i < M; ++i) Y[i] = 0.0f;
        return;
    }
    float s = 127.0f / amax;
    int8_t xq[K];           // VLA: quantized activations
    for (int j = 0; j < K; ++j) {
        long r = lrintf(X[j] * s);
        if (r > 127) r = 127; else if (r < -127) r = -127;
        xq[j] = (int8_t)r;
    }
    int32_t acc[M];         // VLA: int32 accumulation
    ternary_gemv_avx2(acc, W, xq, M, K);
    float k_scale = scale / s;
    for (int i = 0; i < M; ++i) Y[i] = (float)acc[i] * k_scale;
}

// 2-bit packed ternary GEMV. Reads K/4 bytes per row instead of K, so the
// kernel streams 4x fewer weight bytes from DRAM (the actual bottleneck).
// Each byte packs 4 codes (0 -> 0, 1 -> +1, 3 -> -1). The activation plane is
// de-interleaved once per call into 4 residue classes mod 4, so each 2-bit
// field aligns with a contiguous int8 activation vector.
//
// The 2-bit -> {-1,0,1} expansion uses a single vpshufb lookup per field, and
// the row loop is parallelized with OpenMP across the available cores.
#define PACKED_ACCUM_FIELD(SHIFT)                                               \
    do {                                                                        \
        __m256i code = _mm256_and_si256(_mm256_srli_epi16(v, (SHIFT)), three);  \
        __m256i w = _mm256_shuffle_epi8(lut, code);                            \
        __m256i xv = _mm256_loadu_si256(                                        \
            (const __m256i *)(planes[(SHIFT) >> 1] + (size_t)g * 32));          \
        __m256i sa = _mm256_sign_epi8(xv, w);                                   \
        __m256i p16 = _mm256_maddubs_epi16(ones_u8, sa);                        \
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(p16, ones_i16));      \
    } while (0)

void ternary_gemv_packed_f32(float *Y, const uint8_t *Wp, const float *X, int M, int K, float scale) {
    if (K % 4 != 0) {  // row packing would straddle; use the portable reference
        ternary_gemv_packed_ref(Y, Wp, X, M, K, scale);
        return;
    }
    float amax = 0.0f;
    for (int j = 0; j < K; ++j) {
        float a = fabsf(X[j]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-8f) {
        for (int i = 0; i < M; ++i) Y[i] = 0.0f;
        return;
    }
    float s = 127.0f / amax;
    int8_t xq[K];           // VLA: quantized activations
    for (int j = 0; j < K; ++j) {
        long r = lrintf(X[j] * s);
        if (r > 127) r = 127; else if (r < -127) r = -127;
        xq[j] = (int8_t)r;
    }

    int nb = K / 4;                 // packed bytes per row
    int groups = (nb + 31) / 32;    // 32-byte chunks per row
    int plen = groups * 32;         // padded activation-plane length
    int8_t *plane = (int8_t *)calloc((size_t)4 * plen, 1);
    if (!plane) { ternary_gemv_packed_ref(Y, Wp, X, M, K, scale); return; }
    int8_t *p0 = plane, *p1 = plane + plen, *p2 = plane + 2 * plen, *p3 = plane + 3 * plen;
    for (int b = 0; b < nb; ++b) {  // de-interleave activations by residue mod 4
        p0[b] = xq[4 * b + 0];
        p1[b] = xq[4 * b + 1];
        p2[b] = xq[4 * b + 2];
        p3[b] = xq[4 * b + 3];
    }

    const __m256i ones_u8 = _mm256_set1_epi8(1);
    const __m256i ones_i16 = _mm256_set1_epi16(1);
    const __m256i three = _mm256_set1_epi8(3);
    // Byte LUT indexed by a 2-bit code: 0 -> 0, 1 -> +1, 3 -> -1 (replicated into
    // both 128-bit lanes for vpshufb). One shuffle replaces compare/and/sub.
    const __m256i lut = _mm256_setr_epi8(
        0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const int8_t *planes[4] = {p0, p1, p2, p3};
    const float k_scale = scale / s;

    // Parallelize only when the row count amortizes the thread launch.
    #pragma omp parallel for schedule(static) if (M >= 256)
    for (int i = 0; i < M; ++i) {
        const uint8_t *row = Wp + (size_t)i * nb;
        __m256i acc32 = _mm256_setzero_si256();
        for (int g = 0; g < groups; ++g) {
            const uint8_t *src = row + (size_t)g * 32;
            __m256i v;
            int rem = nb - g * 32;
            if (rem >= 32) {
                v = _mm256_loadu_si256((const __m256i *)src);
            } else {
                uint8_t tmp[32];
                memset(tmp, 0, sizeof(tmp));
                memcpy(tmp, src, (size_t)rem);
                v = _mm256_loadu_si256((const __m256i *)tmp);
            }
            // Extract the 2-bit fields. A 16-bit lane shift leaks neighbouring
            // bits into a byte's high part, which the 0x03 mask removes.
            PACKED_ACCUM_FIELD(0);
            PACKED_ACCUM_FIELD(2);
            PACKED_ACCUM_FIELD(4);
            PACKED_ACCUM_FIELD(6);
        }
        __m128i lo = _mm256_castsi256_si128(acc32);
        __m128i hi = _mm256_extracti128_si256(acc32, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        Y[i] = (float)_mm_cvtsi128_si32(sum128) * k_scale;
    }
    free(plane);
}
#undef PACKED_ACCUM_FIELD

void fp32_gemv_par(float *Y, const float *W, const float *X, int M, int K) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        const float *row = W + (size_t)i * K;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= K - 8; j += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + j),
                                  _mm256_loadu_ps(X + j), acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float total = _mm_cvtss_f32(s);
        for (; j < K; ++j) total += row[j] * X[j];
        Y[i] = total;
    }
}

#else

// Fallback when AVX2 is not available
void ternary_gemv_avx2(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K) {
    ternary_gemv_ref(Y, W, X, M, K);
}

void ternary_gemv_f32(float *Y, const int8_t *W, const float *X, int M, int K, float scale) {
    ternary_gemv_f32_ref(Y, W, X, M, K, scale);
}

void ternary_gemv_packed_f32(float *Y, const uint8_t *Wp, const float *X, int M, int K, float scale) {
    ternary_gemv_packed_ref(Y, Wp, X, M, K, scale);
}

void fp32_gemv_par(float *Y, const float *W, const float *X, int M, int K) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        const float *row = W + (size_t)i * K;
        for (int j = 0; j < K; ++j) sum += row[j] * X[j];
        Y[i] = sum;
    }
}

#endif
