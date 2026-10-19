#include "ternary_gemm.h"
#include <string.h>
#include <math.h>

void fp32_gemv_baseline(float *Y, const float *A, const float *X, int M, int K) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        const float *row = A + i * K;
        for (int j = 0; j < K; ++j) {
            sum += row[j] * X[j];
        }
        Y[i] = sum;
    }
}

void int8_gemv_baseline(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K) {
    for (int i = 0; i < M; ++i) {
        int32_t sum = 0;
        const int8_t *row = W + i * K;
        for (int j = 0; j < K; ++j) {
            sum += (int32_t)row[j] * (int32_t)X[j];
        }
        Y[i] = sum;
    }
}

void ternary_gemv_ref(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K) {
    for (int i = 0; i < M; ++i) {
        int32_t sum = 0;
        const int8_t *row = W + i * K;
        for (int j = 0; j < K; ++j) {
            int8_t w = row[j];
            // Zero-multiplication: Pure Addition & Subtraction
            if (w == 1) {
                sum += (int32_t)X[j];
            } else if (w == -1) {
                sum -= (int32_t)X[j];
            }
            // w == 0 is skipped completely
        }
        Y[i] = sum;
    }
}

void ternary_gemv_f32_ref(float *Y, const int8_t *W, const float *X, int M, int K, float scale) {
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
    ternary_gemv_ref(acc, W, xq, M, K);
    float k_scale = scale / s;
    for (int i = 0; i < M; ++i) Y[i] = (float)acc[i] * k_scale;
}

void ternary_gemv_packed_ref(float *Y, const uint8_t *Wp, const float *X, int M, int K, float scale) {
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
    int nb = (K + 3) / 4;   // packed 2-bit bytes per row
    for (int i = 0; i < M; ++i) {
        const uint8_t *row = Wp + (size_t)i * nb;
        int32_t acc = 0;
        for (int b = 0; b < nb; ++b) {
            uint8_t byte = row[b];
            int base = 4 * b;
            for (int t = 0; t < 4; ++t) {
                int j = base + t;
                if (j >= K) break;
                int code = (byte >> (2 * t)) & 0x03;
                if (code == 1) acc += (int32_t)xq[j];
                else if (code == 3) acc -= (int32_t)xq[j];
            }
        }
        Y[i] = (float)acc * (scale / s);
    }
}

void pack_ternary_bitmasks(uint64_t *mask_pos,
                           uint64_t *mask_neg,
                           const int8_t *W,
                           int M, int K) {
    int words_per_row = (K + 63) / 64;
    memset(mask_pos, 0, M * words_per_row * sizeof(uint64_t));
    memset(mask_neg, 0, M * words_per_row * sizeof(uint64_t));

    for (int i = 0; i < M; ++i) {
        const int8_t *row = W + i * K;
        for (int j = 0; j < K; ++j) {
            int word_idx = i * words_per_row + (j / 64);
            int bit_pos = j % 64;
            if (row[j] == 1) {
                mask_pos[word_idx] |= ((uint64_t)1 << bit_pos);
            } else if (row[j] == -1) {
                mask_neg[word_idx] |= ((uint64_t)1 << bit_pos);
            }
        }
    }
}

void ternary_gemv_bitmask(int32_t *Y,
                          const uint64_t *mask_pos,
                          const uint64_t *mask_neg,
                          const int8_t *X,
                          int M, int K) {
    int words_per_row = (K + 63) / 64;

    for (int i = 0; i < M; ++i) {
        int32_t sum = 0;
        const uint64_t *row_pos = mask_pos + i * words_per_row;
        const uint64_t *row_neg = mask_neg + i * words_per_row;

        for (int w = 0; w < words_per_row; ++w) {
            uint64_t p = row_pos[w];
            uint64_t n = row_neg[w];
            int base_j = w * 64;

            // Iterate only over set bits using bit extraction (ctz)
            while (p != 0) {
                int bit = __builtin_ctzll(p);
                int j = base_j + bit;
                if (j < K) {
                    sum += (int32_t)X[j];
                }
                p &= (p - 1); // clear least significant set bit
            }

            while (n != 0) {
                int bit = __builtin_ctzll(n);
                int j = base_j + bit;
                if (j < K) {
                    sum -= (int32_t)X[j];
                }
                n &= (n - 1); // clear least significant set bit
            }
        }
        Y[i] = sum;
    }
}
