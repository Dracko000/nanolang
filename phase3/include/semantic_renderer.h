#ifndef SEMANTIC_RENDERER_H
#define SEMANTIC_RENDERER_H

#include "semantic_dsl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SEM_LANG_INDONESIAN = 0,
    SEM_LANG_ENGLISH    = 1
} semantic_target_lang_t;

/**
 * Deterministically synthesizes natural language text from compact AST tokens.
 * Zero Neural Compute: Runs on local CPU in < 10 microseconds with 0 FLOPs of neural inference.
 */
int semantic_render_text(char *out_buffer,
                         size_t max_buffer_len,
                         const semantic_sequence_t *seq,
                         semantic_target_lang_t lang);

#ifdef __cplusplus
}
#endif

#endif /* SEMANTIC_RENDERER_H */
