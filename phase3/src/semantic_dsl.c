#include "semantic_dsl.h"
#include <string.h>

void semantic_sequence_init(semantic_sequence_t *seq) {
    if (!seq) return;
    memset(seq, 0, sizeof(semantic_sequence_t));
}

int semantic_sequence_append(semantic_sequence_t *seq, uint16_t opcode, int16_t val) {
    if (!seq || seq->count >= MAX_SEMANTIC_TOKENS) return -1;
    seq->tokens[seq->count].opcode = opcode;
    seq->tokens[seq->count].val = val;
    seq->count++;
    return seq->count;
}
