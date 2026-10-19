#ifndef NANOLANG_BRIDGE_H
#define NANOLANG_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* nanolang_handle_t;

/**
 * Initializes the NanoLang mobile runtime by mapping a .nanow file.
 * Returns opaque handle pointer, or NULL on error.
 */
nanolang_handle_t nanolang_mobile_init(const char *model_path);

/**
 * Frees runtime resources and unmaps model.
 */
void nanolang_mobile_free(nanolang_handle_t handle);

/**
 * Executes high-level inference with semantic bytecode decoding.
 * lang_id: 0 = Indonesian, 1 = English.
 * Returns bytes written into response_out.
 */
int nanolang_mobile_generate(nanolang_handle_t handle,
                             const char *prompt,
                             char *response_out,
                             size_t max_out_len,
                             int lang_id);

/**
 * Returns fixed context state RAM footprint in Kilobytes.
 */
float nanolang_mobile_get_state_kb(nanolang_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif /* NANOLANG_BRIDGE_H */
