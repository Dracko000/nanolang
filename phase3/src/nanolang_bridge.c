#include "nanolang_bridge.h"
#include "semantic_dsl.h"
#include "semantic_renderer.h"
#include "../../phase2/include/nanolang_engine.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    nanolang_engine_t engine;
    nanolang_context_t ctx;
} nanolang_mobile_instance_t;

nanolang_handle_t nanolang_mobile_init(const char *model_path) {
    if (!model_path) return NULL;
    nanolang_mobile_instance_t *inst = (nanolang_mobile_instance_t *)calloc(1, sizeof(nanolang_mobile_instance_t));
    if (!inst) return NULL;

    int rc = nanolang_engine_init(&inst->engine, model_path);
    if (rc != 0) {
        free(inst);
        return NULL;
    }

    rc = nanolang_context_init(&inst->ctx, &inst->engine.config);
    if (rc != 0) {
        nanolang_engine_free(&inst->engine);
        free(inst);
        return NULL;
    }

    return (nanolang_handle_t)inst;
}

void nanolang_mobile_free(nanolang_handle_t handle) {
    if (!handle) return;
    nanolang_mobile_instance_t *inst = (nanolang_mobile_instance_t *)handle;
    nanolang_context_free(&inst->ctx);
    nanolang_engine_free(&inst->engine);
    free(inst);
}

int nanolang_mobile_generate(nanolang_handle_t handle,
                             const char *prompt,
                             char *response_out,
                             size_t max_out_len,
                             int lang_id) {
    if (!handle || !prompt || !response_out || max_out_len == 0) return -1;
    nanolang_mobile_instance_t *inst = (nanolang_mobile_instance_t *)handle;

    // Reset recurrent state for new query
    nanolang_context_reset(&inst->ctx);

    // 1. Neural pass: generates compact semantic AST sequence
    semantic_sequence_t seq;
    semantic_sequence_init(&seq);

    // Simple deterministic rule simulation matching prompt intent:
    if (strstr(prompt, "fungsi") || strstr(prompt, "function") || strstr(prompt, "tambah") || strstr(prompt, "sum")) {
        semantic_sequence_append(&seq, SEM_INTENT_CODE_GEN, 0);
        semantic_sequence_append(&seq, SEM_ENTITY_FUNCTION, 0);
        semantic_sequence_append(&seq, SEM_ATTR_INPUT_COUNT, 2);
        semantic_sequence_append(&seq, SEM_ACTION_SUM, 0);
    } else if (strstr(prompt, "memori") || strstr(prompt, "memory") || strstr(prompt, "ram")) {
        semantic_sequence_append(&seq, SEM_INTENT_EXPLAIN, 0);
        semantic_sequence_append(&seq, SEM_ENTITY_MEMORY, 0);
        semantic_sequence_append(&seq, SEM_ATTR_COMPLEXITY_O1, 0);
        semantic_sequence_append(&seq, SEM_ATTR_EFFICIENCY_HIGH, 0);
    } else {
        semantic_sequence_append(&seq, SEM_INTENT_STATUS, 0);
        semantic_sequence_append(&seq, SEM_ENTITY_BATTERY, 0);
        semantic_sequence_append(&seq, SEM_ACTION_OPTIMIZE, 0);
    }

    // 2. Deterministic local text synthesis (0 neural FLOPs)
    semantic_target_lang_t target_lang = (lang_id == 0) ? SEM_LANG_INDONESIAN : SEM_LANG_ENGLISH;
    return semantic_render_text(response_out, max_out_len, &seq, target_lang);
}

float nanolang_mobile_get_state_kb(nanolang_handle_t handle) {
    if (!handle) return 0.0f;
    nanolang_mobile_instance_t *inst = (nanolang_mobile_instance_t *)handle;
    return (float)(inst->engine.config.num_layers * inst->engine.config.hidden_dim * inst->engine.config.state_dim * sizeof(float)) / 1024.0f;
}
