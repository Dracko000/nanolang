#ifndef NANOLANG_ENGINE_H
#define NANOLANG_ENGINE_H

#include "../../phase1/include/nanow_mmap.h"
#include "../../phase1/include/ssm_recurrent.h"
#include "../../phase0/include/ternary_gemm.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t num_layers;
    uint32_t state_dim;
} nanolang_config_t;

typedef struct {
    int num_layers;
    ssm_layer_state_t *layer_states; /* Array of recurrent states, 1 per layer (O(1) memory) */
    
    /* Preallocated reusable forward buffers (Zero heap allocation during generation) */
    float *x;
    float *residual;
    float *normed;
    int8_t *normed_i8;
    int32_t *accum_i32;
    float *ssm_out;
    float *logits;
} nanolang_context_t;

typedef struct {
    nanow_model_t model;
    nanolang_config_t config;
    
    /* Cached Direct Pointers (Zero-Copy) */
    const float *embed_table;               /* vocab_size x hidden_dim */
    const float *final_norm_weight;         /* hidden_dim */
    const float *lm_head_weight;            /* vocab_size x hidden_dim */
    
    /* Per-layer tensor descriptors */
    const nanow_tensor_desc_t **in_proj_descs;
    const nanow_tensor_desc_t **out_proj_descs;
    const float **delta_weights;
    const float **A_weights;
    const float **B_weights;
    const float **C_weights;
    const float **norm_weights;
} nanolang_engine_t;

int nanolang_engine_init(nanolang_engine_t *engine, const char *model_path);
void nanolang_engine_free(nanolang_engine_t *engine);

int nanolang_context_init(nanolang_context_t *ctx, const nanolang_config_t *cfg);
void nanolang_context_free(nanolang_context_t *ctx);
void nanolang_context_reset(nanolang_context_t *ctx);

/**
 * Executes a single autoregressive step: Token ID -> Output Logits.
 * Completely zero-copy on weights, O(1) memory on KV-state, zero-multiplication on linear projections!
 */
int nanolang_forward_step(const nanolang_engine_t *engine,
                         nanolang_context_t *ctx,
                         int token_id,
                         float *logits_out);

/**
 * Samples the next token from logits using Temperature and Top-P (Nucleus) sampling.
 */
int nanolang_sample_token(const float *logits,
                          int vocab_size,
                          float temperature,
                          float top_p,
                          uint32_t *rng_state);

/**
 * Computes Root Mean Square Layer Normalization: y = (x / rms(x)) * alpha
 */
void rmsnorm_forward(float *y, const float *x, const float *alpha, int dim, float eps);

#ifdef __cplusplus
}
#endif

#endif /* NANOLANG_ENGINE_H */
