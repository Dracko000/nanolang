#include "nanolang_engine.h"
#include <math.h>

void rmsnorm_forward(float *y, const float *x, const float *alpha, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / (float)dim + eps);
    float inv_rms = 1.0f / rms;

    if (alpha) {
        for (int i = 0; i < dim; ++i) {
            y[i] = x[i] * inv_rms * alpha[i];
        }
    } else {
        for (int i = 0; i < dim; ++i) {
            y[i] = x[i] * inv_rms;
        }
    }
}
