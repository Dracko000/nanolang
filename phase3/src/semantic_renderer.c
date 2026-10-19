#include "semantic_renderer.h"
#include <stdio.h>
#include <string.h>

int semantic_render_text(char *out_buffer,
                         size_t max_buffer_len,
                         const semantic_sequence_t *seq,
                         semantic_target_lang_t lang) {
    if (!out_buffer || max_buffer_len == 0 || !seq || seq->count == 0) return -1;
    out_buffer[0] = '\0';

    int intent = 0;
    int entity = 0;
    int action = 0;
    int input_count = 0;
    bool has_o1_memory = false;
    bool has_high_efficiency = false;
    bool has_battery = false;

    for (int i = 0; i < seq->count; ++i) {
        uint16_t op = seq->tokens[i].opcode;
        int16_t val = seq->tokens[i].val;

        if (op >= 0x0100 && op < 0x0200) intent = op;
        else if (op >= 0x0200 && op < 0x0300) {
            entity = op;
            if (op == SEM_ENTITY_BATTERY) has_battery = true;
        }
        else if (op == SEM_ATTR_INPUT_COUNT) input_count = val;
        else if (op == SEM_ATTR_COMPLEXITY_O1) has_o1_memory = true;
        else if (op == SEM_ATTR_EFFICIENCY_HIGH) has_high_efficiency = true;
        else if (op >= 0x0400 && op < 0x0500) action = op;
    }

    if (lang == SEM_LANG_INDONESIAN) {
        if (entity == SEM_ENTITY_FUNCTION && action == SEM_ACTION_SUM) {
            snprintf(out_buffer, max_buffer_len,
                     "Fungsi ini menerima %d parameter masukan dan mengembalikan hasil penjumlahannya secara langsung.",
                     input_count > 0 ? input_count : 2);
        } else if (entity == SEM_ENTITY_MEMORY || has_o1_memory) {
            snprintf(out_buffer, max_buffer_len,
                     "Penggunaan memori beroperasi dalam kompleksitas O(1) konstan (bebas KV-cache) dengan efisiensi termal tinggi.");
        } else if (has_battery || entity == SEM_ENTITY_DEVICE) {
            snprintf(out_buffer, max_buffer_len,
                     "Inferensi perangkat berjalan pada profil daya sub-watt rendah (< 2.5W) guna menghemat daya baterai ponsel.");
        } else {
            snprintf(out_buffer, max_buffer_len,
                     "Struktur operasi berhasil dieksekusi secara native melalui model NanoLang di perangkat.");
        }
    } else { // SEM_LANG_ENGLISH
        if (entity == SEM_ENTITY_FUNCTION && action == SEM_ACTION_SUM) {
            snprintf(out_buffer, max_buffer_len,
                     "This function accepts %d input parameters and returns their sum directly.",
                     input_count > 0 ? input_count : 2);
        } else if (entity == SEM_ENTITY_MEMORY || has_o1_memory) {
            snprintf(out_buffer, max_buffer_len,
                     "Memory footprint operates in constant O(1) complexity (zero KV-cache) with high thermal efficiency.");
        } else if (has_battery || entity == SEM_ENTITY_DEVICE) {
            snprintf(out_buffer, max_buffer_len,
                     "Device inference runs at a sub-watt power envelope (< 2.5W) to preserve mobile battery lifespan.");
        } else {
            snprintf(out_buffer, max_buffer_len,
                     "Structural operation successfully executed natively via the on-device NanoLang engine.");
        }
    }

    return (int)strlen(out_buffer);
}
