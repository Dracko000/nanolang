#include "nanolang_qnn.h"
#include <string.h>

void qnn_hexagon_ternary_gemv(int32_t *Y,
                              const uint8_t *packed_W,
                              const int8_t *X,
                              int M, int K) {
    int bytes_per_row = (K + 3) / 4;

    for (int i = 0; i < M; ++i) {
        int32_t acc = 0;
        const uint8_t *row_w = packed_W + i * bytes_per_row;

        // Process in 128-byte vector chunks (512 weights per HVX chunk)
        for (int b = 0; b < bytes_per_row; ++b) {
            uint8_t byte = row_w[b];
            int k_base = b * 4;

            for (int bit = 0; bit < 4; ++bit) {
                int k = k_base + bit;
                if (k < K) {
                    uint8_t code = (byte >> (bit * 2)) & 0x03;
                    if (code == 1) {
                        acc += (int32_t)X[k];
                    } else if (code == 3) {
                        acc -= (int32_t)X[k];
                    }
                }
            }
        }
        Y[i] = acc;
    }
}
