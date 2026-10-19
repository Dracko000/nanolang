#ifndef SEMANTIC_DSL_H
#define SEMANTIC_DSL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_SEMANTIC_TOKENS 128

typedef enum {
    /* Intents */
    SEM_INTENT_EXPLAIN      = 0x0100,
    SEM_INTENT_CODE_GEN     = 0x0101,
    SEM_INTENT_SUMMARY      = 0x0102,
    SEM_INTENT_CALCULATE    = 0x0103,
    SEM_INTENT_STATUS       = 0x0104,

    /* Structural Entities */
    SEM_ENTITY_FUNCTION     = 0x0200,
    SEM_ENTITY_ALGORITHM    = 0x0201,
    SEM_ENTITY_MEMORY       = 0x0202,
    SEM_ENTITY_BATTERY      = 0x0203,
    SEM_ENTITY_DEVICE       = 0x0204,

    /* Attributes / Modifiers */
    SEM_ATTR_INPUT_COUNT    = 0x0300,
    SEM_ATTR_RETURN_TYPE    = 0x0301,
    SEM_ATTR_COMPLEXITY_O1  = 0x0302,
    SEM_ATTR_COMPLEXITY_ON  = 0x0303,
    SEM_ATTR_EFFICIENCY_HIGH= 0x0304,

    /* Action Primitives */
    SEM_ACTION_SUM          = 0x0400,
    SEM_ACTION_TRANSFORM    = 0x0401,
    SEM_ACTION_OPTIMIZE     = 0x0402,
    SEM_ACTION_STREAM       = 0x0403,

    /* Delimiters */
    SEM_NODE_START          = 0x0001,
    SEM_NODE_END            = 0x0002,
    SEM_END_OF_THOUGHT      = 0x0003
} semantic_opcode_t;

typedef struct {
    uint16_t opcode;   /* semantic_opcode_t */
    int16_t  val;      /* numeric argument, e.g. param count = 2 */
} semantic_token_t;

typedef struct {
    semantic_token_t tokens[MAX_SEMANTIC_TOKENS];
    int count;
} semantic_sequence_t;

void semantic_sequence_init(semantic_sequence_t *seq);
int semantic_sequence_append(semantic_sequence_t *seq, uint16_t opcode, int16_t val);

#ifdef __cplusplus
}
#endif

#endif /* SEMANTIC_DSL_H */
