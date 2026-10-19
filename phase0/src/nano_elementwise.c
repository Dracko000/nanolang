/* nano_elementwise.c — transformer elementwise ops in C (AVX2 + scalar).
 *
 * Companion to ternary_avx2.c. The GEMV there is the bandwidth-bound ternary
 * math; these are the O(hidden) and O(seq) ops that sit *between* the GEMMs
 * (RMSNorm, SwiGLU gate, RoPE, softmax). They are cheap in FLOPs but expensive
 * in per-call NumPy/Python overhead on the single-token decode path.
 *
 * The scalar versions of SiLU and softmax lost to NumPy, whose exp() is
 * vectorised. Rather than approximate exp coarsely (which would degrade the
 * numerics the whole project is built on), this file carries a float32 exp
 * built from the standard two-step reduction exp(x) = 2^n * 2^f with a degree-6
 * minimax polynomial, evaluated 8-wide with FMA. Its relative error is ~1e-7,
 * i.e. at the float32 rounding floor, so the kernels stay element-for-element
 * comparable to the NumPy reference.
 *
 * Semantics deliberately match run_smollm2.py's NumPy reference:
 *   rms_norm(x, w)            = x / sqrt(mean(x^2) + eps) * w
 *   silu(g) * up              = g * (1 / (1 + exp(-clip(g, -20, 20)))) * up
 *   apply_rope(x, cos, sin)   = rotate each head's two halves
 *   softmax(scores, axis=1)   = per (kv, group) normalisation over seq
 *
 * Sources: Zhang & Sennrich, "Root Mean Square Layer Normalization"
 * (arXiv:1910.07467); Su et al., "RoFormer: Enhanced Transformer with Rotary
 * Position Embedding" (arXiv:2104.09864); Hendrycks & Gimpel, "Gaussian Error
 * Linear Units" (arXiv:1606.08415) for the SiLU/Swish form; Moshier, "Cephes
 * Mathematical Library", for the minimax exp polynomial coefficients.
 * The SIMD build is only compiled when the translation unit sees __AVX2__; the
 * scalar path stays valid for the ARM/NEON and NPU backends.
 */
#include <math.h>
#include <stddef.h>

#include "ternary_gemm.h"

#if defined(__AVX2__)
#include <immintrin.h>

/* exp(x) for 8 lanes. Reduces x = n*ln2 + r with |r| <= ln2/2, evaluates a
 * degree-6 polynomial on r, then scales by 2^n built from the exponent bits.
 * Relative error ~1e-7 across the float32 range. */
static inline __m256 exp_ps(__m256 x) {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 p0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 p1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 p2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 p3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 p4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 p5 = _mm256_set1_ps(5.0000001201E-1f);
    __m256 fx, tmp, mask, z, y, pow2n;

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    fx = _mm256_mul_ps(x, log2e);
    fx = _mm256_add_ps(fx, half);
    tmp = _mm256_floor_ps(fx);
    mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, c1);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c2));

    z = _mm256_mul_ps(x, x);
    y = p0;
    y = _mm256_fmadd_ps(y, x, p1);
    y = _mm256_fmadd_ps(y, x, p2);
    y = _mm256_fmadd_ps(y, x, p3);
    y = _mm256_fmadd_ps(y, x, p4);
    y = _mm256_fmadd_ps(y, x, p5);
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(0x7f)), 23));
    return _mm256_mul_ps(y, pow2n);
}

static inline float hmax_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x4E));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0xB1));
    return _mm_cvtss_f32(lo);
}

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif /* __AVX2__ */

static inline float exp_clip_scalar(float x) {
    float c = x < -20.0f ? -20.0f : (x > 20.0f ? 20.0f : x);
    return expf(-c);
}

void nano_rmsnorm_f32(float *out, const float *x, const float *w, int n, float eps) {
    /* Sum of squares in double so the reduction is stable for long rows. */
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        double v = (double)x[i];
        ss += v * v;
    }
    float rms = (float)sqrt(ss / (double)n + (double)eps);
    for (int i = 0; i < n; i++) {
        out[i] = (x[i] / rms) * w[i];
    }
}

void nano_silu_mul_f32(float *out, const float *gate, const float *up, int n) {
    int i = 0;
#if defined(__AVX2__)
    const __m256 vmin = _mm256_set1_ps(-20.0f);
    const __m256 vmax = _mm256_set1_ps(20.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 gc = _mm256_min_ps(_mm256_max_ps(g, vmin), vmax);
        __m256 e = exp_ps(_mm256_sub_ps(zero, gc));           /* exp(-clip(g)) */
        __m256 inv = _mm256_div_ps(one, _mm256_add_ps(one, e));
        __m256 s = _mm256_mul_ps(g, inv);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(s, _mm256_loadu_ps(up + i)));
    }
#endif
    for (; i < n; i++) {
        float g = gate[i];
        out[i] = (g * (1.0f / (1.0f + exp_clip_scalar(g)))) * up[i];
    }
}

void nano_rope_f32(float *x, const float *cos, const float *sin,
                   int n_heads, int head_dim) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *base = x + (size_t)h * (size_t)head_dim;
        for (int i = 0; i < half; i++) {
            float a = base[i];            /* first-half component */
            float b = base[i + half];     /* second-half component */
            float c = cos[i];
            float s = sin[i];
            base[i] = a * c - b * s;
            base[i + half] = b * c + a * s;
        }
    }
}

/* Row-wise softmax over ``rows`` contiguous rows of length ``n``. */
static void softmax_rows_f32(float *x, int rows, int n) {
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * (size_t)n;
        int i = 0;
        float m = row[0];
        double sum = 0.0;
#if defined(__AVX2__)
        if (n >= 8) {
            __m256 vm = _mm256_loadu_ps(row);
            for (i = 8; i + 8 <= n; i += 8) {
                vm = _mm256_max_ps(vm, _mm256_loadu_ps(row + i));
            }
            m = hmax_ps(vm);
        }
#endif
        for (; i < n; i++) {
            if (row[i] > m) m = row[i];
        }
        i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= n; i += 8) {
            __m256 e = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + i),
                                            _mm256_set1_ps(m)));
            _mm256_storeu_ps(row + i, e);
            sum += (double)hsum_ps(e);
        }
#endif
        for (; i < n; i++) {
            float e = expf(row[i] - m);
            row[i] = e;
            sum += (double)e;
        }
        float inv = (float)(1.0 / sum);
        i = 0;
#if defined(__AVX2__)
        __m256 vinv = _mm256_set1_ps(inv);
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(row + i,
                             _mm256_mul_ps(_mm256_loadu_ps(row + i), vinv));
        }
#endif
        for (; i < n; i++) {
            row[i] *= inv;
        }
    }
}

void nano_softmax_axis1_f32(float *x, int axis0, int n, int axis2) {
    /* The attention scores are (n_kv, seq, n_groups); with n_groups == 1 (as in
     * BitNet 700M, 16 heads / 16 kv heads) each row is contiguous, which is the
     * fast vectorised path. Other shapes fall back to the scalar strided walk. */
    if (axis2 == 1) {
        softmax_rows_f32(x, axis0, n);
        return;
    }
    for (int a = 0; a < axis0; a++) {
        for (int g = 0; g < axis2; g++) {
            float *base = x + ((size_t)a * (size_t)n) * (size_t)axis2 + (size_t)g;
            float m = base[0];
            for (int i = 1; i < n; i++) {
                float v = base[(size_t)i * (size_t)axis2];
                if (v > m) m = v;
            }
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                float e = expf(base[(size_t)i * (size_t)axis2] - m);
                base[(size_t)i * (size_t)axis2] = e;
                sum += (double)e;
            }
            float inv = (float)(1.0 / sum);
            for (int i = 0; i < n; i++) {
                base[(size_t)i * (size_t)axis2] *= inv;
            }
        }
    }
}
