#include "ternary_gemm.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

void ternary_gemv_neon(int32_t *Y, const int8_t *W, const int8_t *X, int M, int K) {
    const int8x16_t vzero = vdupq_n_s8(0);

    for (int i = 0; i < M; ++i) {
        const int8_t *row = W + i * K;
        int32x4_t vacc0 = vdupq_n_s32(0);
        int32x4_t vacc1 = vdupq_n_s32(0);
        int j = 0;

        for (; j <= K - 16; j += 16) {
            int8x16_t vx = vld1q_s8(X + j);
            int8x16_t vw = vld1q_s8(row + j);

            // Create masks for positive (+1) and negative (-1) weights
            uint8x16_t mask_pos = vcgtq_s8(vw, vzero);
            uint8x16_t mask_neg = vcltq_s8(vw, vzero);

            // Masked activations:
            // vpos contains X where W == +1, 0 otherwise
            // vneg contains X where W == -1, 0 otherwise
            int8x16_t vpos = vandq_s8(vx, (int8x16_t)mask_pos);
            int8x16_t vneg = vandq_s8(vx, (int8x16_t)mask_neg);

            // Subtract vneg from vpos (pure vector addition and subtraction)
            int8x16_t vdiff = vsubq_s8(vpos, vneg);

            // Widen and accumulate into 32-bit registers
            int16x8_t vwide_lo = vmovl_s8(vget_low_s8(vdiff));
            int16x8_t vwide_hi = vmovl_s8(vget_high_s8(vdiff));

            vacc0 = vaddw_s16(vacc0, vget_low_s16(vwide_lo));
            vacc1 = vaddw_s16(vacc1, vget_high_s16(vwide_lo));
            vacc0 = vaddw_s16(vacc0, vget_low_s16(vwide_hi));
            vacc1 = vaddw_s16(vacc1, vget_high_s16(vwide_hi));
        }

        // Horizontal sum of NEON registers
        int32x4_t vacc = vaddq_s32(vacc0, vacc1);
        int32_t total = vaddvq_s32(vacc);

        // Process residual tail scalar
        for (; j < K; ++j) {
            int8_t w = row[j];
            if (w == 1) total += (int32_t)X[j];
            else if (w == -1) total -= (int32_t)X[j];
        }

        Y[i] = total;
    }
}
#endif
