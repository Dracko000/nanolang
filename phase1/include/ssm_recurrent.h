#ifndef SSM_RECURRENT_H
#define SSM_RECURRENT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Recurrent State Space context for a single layer.
 * Invariant Memory O(1): Remains exactly fixed regardless of dialogue length!
 */
typedef struct {
    int hidden_dim;     /* D: e.g. 2048 */
    int state_dim;      /* N: e.g. 16 */
    float *h;           /* Recurrent state matrix: D x N elements (continuous in memory) */
} ssm_layer_state_t;

/**
 * Initializes a recurrent state buffer with zeros.
 */
int ssm_layer_state_init(ssm_layer_state_t *state, int hidden_dim, int state_dim);

/**
 * Frees recurrent state buffer.
 */
void ssm_layer_state_free(ssm_layer_state_t *state);

/**
 * Resets recurrent state to zeros (e.g. start of new conversation session).
 */
void ssm_layer_state_reset(ssm_layer_state_t *state);

/**
 * Executes one autoregressive decode step for Selective State Space Model.
 *
 * Continuous formulation discretized via Euler/ZOH:
 *   A_bar = exp(delta * A)   [or approx 1 + delta * A]
 *   B_bar = delta * B
 *   h_t = A_bar * h_{t-1} + B_bar * x_t
 *   y_t = C * h_t + D * x_t
 *
 * Runs in O(D * N) time, O(1) memory!
 */
void ssm_step_forward(float *y_out,
                      ssm_layer_state_t *state,
                      const float *x_in,
                      const float *delta,
                      const float *A,
                      const float *B,
                      const float *C,
                      const float *D_param);

#ifdef __cplusplus
}
#endif

#endif /* SSM_RECURRENT_H */
