#include "ssm_recurrent.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int ssm_layer_state_init(ssm_layer_state_t *state, int hidden_dim, int state_dim) {
    if (!state || hidden_dim <= 0 || state_dim <= 0) return -1;
    state->hidden_dim = hidden_dim;
    state->state_dim = state_dim;
    state->h = (float *)calloc((size_t)hidden_dim * (size_t)state_dim, sizeof(float));
    if (!state->h) return -2;
    return 0;
}

void ssm_layer_state_free(ssm_layer_state_t *state) {
    if (!state) return;
    if (state->h) {
        free(state->h);
        state->h = NULL;
    }
    state->hidden_dim = 0;
    state->state_dim = 0;
}

void ssm_layer_state_reset(ssm_layer_state_t *state) {
    if (!state || !state->h) return;
    memset(state->h, 0, (size_t)state->hidden_dim * (size_t)state->state_dim * sizeof(float));
}

void ssm_step_forward(float *y_out,
                      ssm_layer_state_t *state,
                      const float *x_in,
                      const float *delta,
                      const float *A,
                      const float *B,
                      const float *C,
                      const float *D_param) {
    int D = state->hidden_dim;
    int N = state->state_dim;

    for (int d = 0; d < D; ++d) {
        float x_val = x_in[d];
        float dt = delta[d];
        // Discretization parameter (ensure positive dt)
        if (dt < 0.0f) dt = 0.0f;

        float y_acc = 0.0f;
        float *h_d = state->h + d * N;

        for (int n = 0; n < N; ++n) {
            float a_val = A[d * N + n];
            // Discretize continuous state matrix A: A_bar = exp(dt * A)
            float a_bar = expf(dt * a_val);
            // Discretize continuous input matrix B: B_bar = dt * B
            float b_bar = dt * B[n];

            // Recurrent state transition: h_t = A_bar * h_{t-1} + B_bar * x_t
            float h_new = a_bar * h_d[n] + b_bar * x_val;
            h_d[n] = h_new;

            // Output projection: y = sum(C * h)
            y_acc += C[n] * h_new;
        }

        // Direct skip connection: y = C*h + D*x
        if (D_param) {
            y_acc += D_param[d] * x_val;
        }

        y_out[d] = y_acc;
    }
}
