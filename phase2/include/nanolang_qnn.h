#ifndef NANOLANG_QNN_H
#define NANOLANG_QNN_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HEXAGON_VECTOR_WIDTH_BYTES 128 /* 1024-bit vector registers on Snapdragon Hexagon */

typedef struct {
    uint32_t num_hvx_units;
    uint32_t vtcm_size_kb;      /* Vector Tightly-Coupled Memory (on-chip SRAM) */
    bool supports_htp_fp16;
    bool supports_htp_int8;
} qnn_hexagon_profile_t;

/**
 * Simulates Qualcomm Hexagon HTP 1024-bit SIMD vector execution
 * for 2-bit ternary linear operations.
 */
void qnn_hexagon_ternary_gemv(int32_t *Y,
                              const uint8_t *packed_W,
                              const int8_t *X,
                              int M, int K);

#ifdef __cplusplus
}
#endif

#endif /* NANOLANG_QNN_H */
