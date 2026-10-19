#ifndef NANO_IR_H
#define NANO_IR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NANO_OP_EMBEDDING_LOOKUP = 0,
    NANO_OP_RMSNORM          = 1,
    NANO_OP_TERNARY_LINEAR   = 2, /* Zero-Mul BitNet linear projection */
    NANO_OP_SSM_STEP         = 3, /* O(1) Memory Recurrent SSM step */
    NANO_OP_SILU_MUL         = 4, /* Gated activation (SwiGLU) */
    NANO_OP_RESIDUAL_ADD     = 5  /* Skip connection addition */
} nano_ir_opcode_t;

typedef struct {
    nano_ir_opcode_t opcode;
    char name[48];
    int input_ids[3];
    int output_id;
    int dim_in;
    int dim_out;
    float scale;
    bool is_ternary;
} nano_ir_node_t;

typedef struct {
    int num_nodes;
    int max_nodes;
    nano_ir_node_t *nodes;
    int hidden_dim;
    int state_dim;
    int num_layers;
} nano_ir_graph_t;

int nano_ir_graph_init(nano_ir_graph_t *graph, int max_nodes, int hidden_dim, int state_dim, int num_layers);
void nano_ir_graph_free(nano_ir_graph_t *graph);
int nano_ir_add_node(nano_ir_graph_t *graph, nano_ir_node_t node);
bool nano_ir_validate_graph(const nano_ir_graph_t *graph, char *err_buf, size_t err_buf_len);
void nano_ir_dump_text(const nano_ir_graph_t *graph);

#ifdef __cplusplus
}
#endif

#endif /* NANO_IR_H */
