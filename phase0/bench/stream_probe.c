/*
 * stream_probe.c — which limit does a GEMV actually hit on this machine?
 *
 * T1 of the roadmap. A GEMV reads every weight byte exactly once per call, so
 * its runtime is bounded below by (bytes / achievable bandwidth) and above by
 * (work / achievable compute). Timing the GEMV alone cannot say which bound is
 * binding, because both produce one number.
 *
 * The trick used here is to measure the *marginal cost of one row* twice:
 *
 *   warm   matrix small enough to stay in L2/L3. Memory is free, so the cost
 *          is the kernel's issue/ALU limit.
 *   cold   matrix far larger than L3. Every row comes from DRAM, so the cost
 *          is at least the DRAM cost of that row's bytes.
 *
 * If warm and cold agree, the kernel is compute-bound and the DRAM stream is
 * already faster than the ALU: shrinking the payload buys nothing.
 * If cold is much slower, the kernel is bandwidth-bound and fewer bytes is the
 * only lever that helps.
 *
 * A single intercept+slope fit isolates per-call overhead (the intercept, from
 * x-quantisation and its temporary allocation) from the per-row work.
 *
 * Build (bench_roofline.py does this):
 *   cc -O3 -mavx2 -mfma -fopenmp stream_probe.c -L phase0/bin -lternary -lm
 *
 * See ROADMAP_IMPLEMENTASI.md §7.10 (T1).
 */

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/ternary_gemm.h"

#define TRI_BYTES_PER_ROW(k) (((k) + 3) / 4)
#define K_HIDDEN 1536

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *alloc64(size_t bytes) {
    size_t rounded = (bytes + 63u) & ~(size_t)63u;
    if (rounded == 0) rounded = 64;
    return aligned_alloc(64, rounded);
}

static void fill(unsigned char *p, size_t n, unsigned seed) {
    unsigned s = seed | 1u;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        unsigned long long v = ((unsigned long long)s << 32) | (unsigned)s;
        memcpy(p + i, &v, 8);
    }
    for (; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (unsigned char)(s & 0xff);
    }
}

/* --- achievable streaming bandwidth, far larger than L3 -------------------- */
static double stream_gbs(const void *buf, size_t bytes, int reps) {
    const __m256i *q = (const __m256i *)buf;
    size_t nv = bytes / 32u;
    double best = 1e30;

    for (int r = 0; r < reps; ++r) {
        __m256i acc = _mm256_setzero_si256();
        double t0 = now_sec();
        #pragma omp parallel
        {
            __m256i local = _mm256_setzero_si256();
            #pragma omp for schedule(static)
            for (size_t i = 0; i < nv; ++i) {
                local = _mm256_xor_si256(local, _mm256_loadu_si256(q + i));
            }
            #pragma omp critical
            acc = _mm256_xor_si256(acc, local);
        }
        double dt = now_sec() - t0;
        if ((unsigned)_mm256_extract_epi32(acc, 0) == 0xdeadbeefu) {
            fprintf(stderr, "unreachable sink\n");   /* keeps acc live */
        }
        if (dt < best) best = dt;
    }
    return (double)(nv * 32u) / best / 1e9;
}

/* --- best-of-N timing of one GEMV call ------------------------------------ */
static double best_call_fp32(const float *W, const float *x, float *y,
                             int M, int K, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_sec();
        fp32_gemv_par(y, W, x, M, K);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static double best_call_tri(const unsigned char *Wp, const float *x, float *y,
                            int M, int K, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_sec();
        ternary_gemv_packed_f32(y, Wp, x, M, K, 1.0f);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* Least-squares fit t(M) = a + b*M over the measured row counts.
 * b is the marginal cost of one row, a the fixed per-call cost. */
typedef struct { double a_us, b_ns, r2; } fit_t;

static fit_t fit(const int *M, const double *t_us, int n) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (int i = 0; i < n; ++i) {
        sx += M[i]; sy += t_us[i];
        sxx += (double)M[i] * M[i];
        sxy += (double)M[i] * t_us[i];
        syy += t_us[i] * t_us[i];
    }
    double den = n * sxx - sx * sx;
    double b = (n * sxy - sx * sy) / den;              /* us per row */
    double a = (sy - b * sx) / n;                      /* us per call */
    double num = n * sxy - sx * sy;
    double d1 = n * sxx - sx * sx, d2 = n * syy - sy * sy;
    double r2 = (d1 > 0 && d2 > 0) ? (num * num) / (d1 * d2) : 0.0;
    fit_t f = { a, b * 1e3, r2 };
    return f;
}

static void measure(const char *label, int K, int m_lo, int m_hi, int m_step,
                    int cold_M, int reps, int ternary) {
    size_t max_rows = (size_t)(cold_M > m_hi ? cold_M : m_hi);
    size_t row_bytes = ternary ? (size_t)TRI_BYTES_PER_ROW(K) : (size_t)K * 4u;
    size_t bytes = max_rows * row_bytes;

    unsigned char *W = (unsigned char *)alloc64(bytes);
    float *y = (float *)alloc64(max_rows * sizeof(float));
    float *x = (float *)alloc64((size_t)K * sizeof(float));
    if (!W || !y || !x) { fprintf(stderr, "oom %s\n", label); exit(1); }
    fill(W, bytes, ternary ? 11 : 7);
    for (int i = 0; i < K; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.03125f;

    /* warm-up so the first timed call never pays page faults or a cold I-cache */
    if (ternary) ternary_gemv_packed_f32(y, W, x, 512, K, 1.0f);
    else         fp32_gemv_par(y, (float *)W, x, 512, K);

    int Ms[64];
    double ts[64];
    int n = 0;
    for (int M = m_lo; M <= m_hi && n < 64; M += m_step) {
        if (ternary) ts[n] = best_call_tri(W, x, y, M, K, reps) * 1e6;
        else         ts[n] = best_call_fp32((float *)W, x, y, M, K, reps) * 1e6;
        Ms[n] = M;
        ++n;
    }
    fit_t w = fit(Ms, ts, n);

    printf("  %-8s row = %zu B   points (rows -> us):", label, row_bytes);
    for (int i = 0; i < n; ++i) printf(" %d->%.2f", Ms[i], ts[i]);
    printf("\n");

    double cold_us = ternary ? best_call_tri(W, x, y, cold_M, K, reps) * 1e6
                             : best_call_fp32((float *)W, x, y, cold_M, K, reps) * 1e6;
    double cold_ns = cold_us / cold_M * 1e3;

    printf("  %-8s warm fit: %6.2f us fixed + %6.1f ns/row  (R2=%.4f)\n",
           "", w.a_us, w.b_ns, w.r2);
    printf("  %-8s cold     : %6.1f ns/row at %d rows (%.1f MB)\n\n",
           "", cold_ns, cold_M, (double)cold_M * row_bytes / 1e6);

    free(W); free(y); free(x);
}

/* --- fixed cost per call vs K -------------------------------------------
 * The fit's intercept is ~9.5 us at K=1536. If that is X-quantisation and its
 * scratch planes it scales with K; if it is the OpenMP thread-team start it
 * does not. The answer decides whether hoisting the scratch buffer helps. */
static void fixed_cost_vs_k(void) {
    const int Ks[] = {384, 768, 1536, 3072, 6144};
    const int Ms[] = {256, 512, 768, 1024};

    printf("\nD. fixed cost per call vs K (ternary, warm, M=256..1024)\n");
    printf("   %6s %10s %12s %10s\n", "K", "fixed us", "ns/row", "R2");
    for (int s = 0; s < (int)(sizeof(Ks) / sizeof(Ks[0])); ++s) {
        int K = Ks[s];
        size_t bytes = (size_t)1024 * TRI_BYTES_PER_ROW(K);
        unsigned char *W = (unsigned char *)alloc64(bytes);
        float *y = (float *)alloc64(1024 * sizeof(float));
        float *x = (float *)alloc64((size_t)K * sizeof(float));
        if (!W || !y || !x) { fprintf(stderr, "oom K\n"); exit(1); }
        fill(W, bytes, 17);
        for (int i = 0; i < K; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.03125f;

        double ts[4];
        for (int i = 0; i < 4; ++i) ts[i] = best_call_tri(W, x, y, Ms[i], K, 15) * 1e6;
        fit_t f = fit(Ms, ts, 4);
        printf("   %6d %10.2f %12.1f %10.4f\n", K, f.a_us, f.b_ns, f.r2);
        free(W); free(y); free(x);
    }
}

int main(void) {
    const int K = K_HIDDEN;
    printf("stream_probe — read floor vs GEMV, K=%d, threads=%d\n\n",
           K, omp_get_max_threads());

    size_t big = 256u << 20;
    unsigned char *buf = (unsigned char *)alloc64(big);
    if (!buf) { fprintf(stderr, "oom\n"); return 1; }
    fill(buf, big, 12345);
    double gbs = stream_gbs(buf, big, 12);
    free(buf);
    printf("A. streaming ceiling: parallel read of %zu MB = %.2f GB/s\n",
           big >> 20, gbs);

    double tri_row_ns = TRI_BYTES_PER_ROW(K) / gbs;      /* bytes / (B/ns) */
    double fp32_row_ns = (double)K * 4.0 / gbs;
    printf("   implied DRAM cost: ternary row %.1f ns, fp32 row %.1f ns\n\n",
           tri_row_ns, fp32_row_ns);

    /* warm: cache-resident, so the per-row cost is the kernel's own limit.
     * cold: far past L3, so every row has to come from DRAM.
     * ternary: 512..4096 rows = 0.2..1.6 MB, all above the kernel's
     *   OpenMP-parallel threshold (M >= 256) and all L2/L3-resident.
     *   cold at 100000 rows = 38.4 MB.
     * fp32: 128..640 rows = 0.8..3.9 MB, inside the 8 MB L3.
     *   cold at 20000 rows = 122.9 MB. */
    printf("B. marginal cost of one row\n");
    measure("ternary", K, 512, 4096, 512, 100000, 15, 1);
    measure("fp32", K, 128, 640, 128, 20000, 11, 0);

    printf("\nC. per-token traffic at M=1 (from the release container)\n");
    {
        int layers = 24, M_ffn = 4096, M_qo = 1536, vocab = 32002;
        double tri = layers * (4.0 * M_qo * TRI_BYTES_PER_ROW(K)
                             + 2.0 * M_ffn * TRI_BYTES_PER_ROW(K)
                             + (double)K * TRI_BYTES_PER_ROW(M_ffn));
        double head = (double)vocab * K * 4.0;
        printf("   ternary projections %7.1f MB  %5.1f%%\n",
               tri / 1e6, 100.0 * tri / (tri + head));
        printf("   tied embed as head  %7.1f MB  %5.1f%%\n",
               head / 1e6, 100.0 * head / (tri + head));
        printf("   total               %7.1f MB\n", (tri + head) / 1e6);
        printf("   at %.2f GB/s that is %.1f ms/token = %.0f tok/s\n",
               gbs, (tri + head) / gbs / 1e6,
               1e9 * gbs / (tri + head));
    }

    fixed_cost_vs_k();
    return 0;
}
