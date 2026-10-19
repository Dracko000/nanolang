#include "ternary_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Generate deterministic pseudo-random ternary weights {-1, 0, +1}
static void generate_ternary_weights(int8_t *W, int size, uint32_t *seed) {
    for (int i = 0; i < size; ++i) {
        *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
        int r = (*seed) % 3; // 0, 1, 2
        if (r == 0) W[i] = -1;
        else if (r == 1) W[i] = 0;
        else W[i] = 1;
    }
}

// Generate deterministic INT8 activations [-128, 127]
static void generate_int8_activations(int8_t *X, int size, uint32_t *seed) {
    for (int i = 0; i < size; ++i) {
        *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
        X[i] = (int8_t)((*seed % 255) - 127);
    }
}

static void run_benchmark_for_size(int M, int K, int iterations) {
    printf("\n======================================================================\n");
    printf(" BENCHMARK LAYER SIZE: M=%d, K=%d (Weights: %d elements, Iterations: %d)\n",
           M, K, M * K, iterations);
    printf("======================================================================\n");

    uint32_t seed = 42;
    int weight_count = M * K;

    // Allocate buffers
    int8_t *W_int8 = (int8_t *)malloc(weight_count * sizeof(int8_t));
    float *W_fp32 = (float *)malloc(weight_count * sizeof(float));
    int8_t *X_int8 = (int8_t *)malloc(K * sizeof(int8_t));
    float *X_fp32 = (float *)malloc(K * sizeof(float));

    int32_t *Y_ref = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *Y_base_int8 = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *Y_bitmask = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *Y_avx2 = (int32_t *)malloc(M * sizeof(int32_t));
    float *Y_fp32 = (float *)malloc(M * sizeof(float));

    int words_per_row = (K + 63) / 64;
    uint64_t *mask_pos = (uint64_t *)malloc(M * words_per_row * sizeof(uint64_t));
    uint64_t *mask_neg = (uint64_t *)malloc(M * words_per_row * sizeof(uint64_t));

    // Initialize data
    generate_ternary_weights(W_int8, weight_count, &seed);
    generate_int8_activations(X_int8, K, &seed);

    for (int i = 0; i < weight_count; ++i) {
        W_fp32[i] = (float)W_int8[i];
    }
    for (int j = 0; j < K; ++j) {
        X_fp32[j] = (float)X_int8[j];
    }

    pack_ternary_bitmasks(mask_pos, mask_neg, W_int8, M, K);

    // 1. Correctness Validation
    ternary_gemv_ref(Y_ref, W_int8, X_int8, M, K);
    int8_gemv_baseline(Y_base_int8, W_int8, X_int8, M, K);
    ternary_gemv_bitmask(Y_bitmask, mask_pos, mask_neg, X_int8, M, K);
    ternary_gemv_avx2(Y_avx2, W_int8, X_int8, M, K);
    fp32_gemv_baseline(Y_fp32, W_fp32, X_fp32, M, K);

    int err_base = 0, err_bitmask = 0, err_avx2 = 0, err_fp32 = 0;
    for (int i = 0; i < M; ++i) {
        if (Y_ref[i] != Y_base_int8[i]) err_base++;
        if (Y_ref[i] != Y_bitmask[i]) err_bitmask++;
        if (Y_ref[i] != Y_avx2[i]) err_avx2++;
        if (fabsf((float)Y_ref[i] - Y_fp32[i]) > 1e-3f) err_fp32++;
    }

    printf("  [Verification] Reference vs INT8 Mul:   %s\n", err_base == 0 ? "PASSED (100% Match)" : "FAILED");
    printf("  [Verification] Reference vs Bitmask:    %s\n", err_bitmask == 0 ? "PASSED (100% Match)" : "FAILED");
    printf("  [Verification] Reference vs AVX2 Add:   %s\n", err_avx2 == 0 ? "PASSED (100% Match)" : "FAILED");
    printf("  [Verification] Reference vs FP32 Baseline: %s\n", err_fp32 == 0 ? "PASSED (100% Match)" : "FAILED");

    // 2. Timed Benchmarks
    double t0, t1;
    double ops = 2.0 * (double)M * (double)K * (double)iterations;

    // --- FP32 Baseline ---
    t0 = get_time_sec();
    for (int it = 0; it < iterations; ++it) {
        fp32_gemv_baseline(Y_fp32, W_fp32, X_fp32, M, K);
    }
    t1 = get_time_sec();
    double time_fp32 = (t1 - t0) / iterations;
    double gops_fp32 = (ops / (t1 - t0)) * 1e-9;

    // --- INT8 Mul Baseline ---
    t0 = get_time_sec();
    for (int it = 0; it < iterations; ++it) {
        int8_gemv_baseline(Y_base_int8, W_int8, X_int8, M, K);
    }
    t1 = get_time_sec();
    double time_int8_mul = (t1 - t0) / iterations;
    double gops_int8_mul = (ops / (t1 - t0)) * 1e-9;

    // --- Ternary Ref (Add/Sub with Zero-Skip) ---
    t0 = get_time_sec();
    for (int it = 0; it < iterations; ++it) {
        ternary_gemv_ref(Y_ref, W_int8, X_int8, M, K);
    }
    t1 = get_time_sec();
    double time_ternary_ref = (t1 - t0) / iterations;
    double gops_ternary_ref = (ops / (t1 - t0)) * 1e-9;

    // --- Ternary Bitmask (2-bit weight footprint) ---
    t0 = get_time_sec();
    for (int it = 0; it < iterations; ++it) {
        ternary_gemv_bitmask(Y_bitmask, mask_pos, mask_neg, X_int8, M, K);
    }
    t1 = get_time_sec();
    double time_bitmask = (t1 - t0) / iterations;
    double gops_bitmask = (ops / (t1 - t0)) * 1e-9;

    // --- Ternary AVX2 (Vectorized Zero-Mul) ---
    t0 = get_time_sec();
    for (int it = 0; it < iterations; ++it) {
        ternary_gemv_avx2(Y_avx2, W_int8, X_int8, M, K);
    }
    t1 = get_time_sec();
    double time_avx2 = (t1 - t0) / iterations;
    double gops_avx2 = (ops / (t1 - t0)) * 1e-9;

    // Report Summary
    printf("\n  Latency per forward token:\n");
    printf("    1. FP32 Baseline:       %8.3f ms  (%6.2f GOPS)  [Weight Size: %5.2f MB]\n",
           time_fp32 * 1000.0, gops_fp32, (float)(weight_count * 4) / (1024.0f * 1024.0f));
    printf("    2. INT8 Mul Baseline:   %8.3f ms  (%6.2f GOPS)  [Weight Size: %5.2f MB]\n",
           time_int8_mul * 1000.0, gops_int8_mul, (float)(weight_count * 1) / (1024.0f * 1024.0f));
    printf("    3. Ternary Ref (Add/Sub):%7.3f ms  (%6.2f GOPS)  [Weight Size: %5.2f MB]\n",
           time_ternary_ref * 1000.0, gops_ternary_ref, (float)(weight_count * 1) / (1024.0f * 1024.0f));
    printf("    4. Ternary Bitmask:     %8.3f ms  (%6.2f GOPS)  [Weight Size: %5.2f MB (2-bit)]\n",
           time_bitmask * 1000.0, gops_bitmask, (float)(M * words_per_row * 16) / (1024.0f * 1024.0f));
    printf("    5. Ternary AVX2 SIMD:   %8.3f ms  (%6.2f GOPS)  [Speedup: %5.2fx vs FP32]\n",
           time_avx2 * 1000.0, gops_avx2, time_fp32 / time_avx2);

    // Free buffers
    free(W_int8);
    free(W_fp32);
    free(X_int8);
    free(X_fp32);
    free(Y_ref);
    free(Y_base_int8);
    free(Y_bitmask);
    free(Y_avx2);
    free(Y_fp32);
    free(mask_pos);
    free(mask_neg);
}

int main(void) {
    printf("======================================================================\n");
    printf(" NANOLANG PHASE 0: MICRO-KERNEL POC & BENCHMARK\n");
    printf(" Zero-Multiplication Ternary GEMV vs Baseline Floating-Point & INT8\n");
    printf("======================================================================\n");

    // Test with typical LLM projection shapes
    run_benchmark_for_size(2048, 2048, 100);  // ~4.1M weights
    run_benchmark_for_size(4096, 4096, 50);   // ~16.7M weights (Llama-3 8B projection block)
    run_benchmark_for_size(8192, 4096, 25);   // ~33.5M weights (Wide MLP layer)

    return 0;
}
