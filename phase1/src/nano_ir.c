#include "nano_ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int nano_ir_graph_init(nano_ir_graph_t *graph, int max_nodes, int hidden_dim, int state_dim, int num_layers) {
    if (!graph || max_nodes <= 0) return -1;
    graph->max_nodes = max_nodes;
    graph->num_nodes = 0;
    graph->hidden_dim = hidden_dim;
    graph->state_dim = state_dim;
    graph->num_layers = num_layers;
    graph->nodes = (nano_ir_node_t *)malloc(sizeof(nano_ir_node_t) * max_nodes);
    if (!graph->nodes) return -2;
    return 0;
}

void nano_ir_graph_free(nano_ir_graph_t *graph) {
    if (!graph) return;
    if (graph->nodes) {
        free(graph->nodes);
        graph->nodes = NULL;
    }
    graph->num_nodes = 0;
    graph->max_nodes = 0;
}

int nano_ir_add_node(nano_ir_graph_t *graph, nano_ir_node_t node) {
    if (!graph || graph->num_nodes >= graph->max_nodes) return -1;
    graph->nodes[graph->num_nodes++] = node;
    return graph->num_nodes - 1;
}

bool nano_ir_validate_graph(const nano_ir_graph_t *graph, char *err_buf, size_t err_buf_len) {
    if (!graph || graph->num_nodes == 0) {
        snprintf(err_buf, err_buf_len, "Graph is empty");
        return false;
    }

    for (int i = 0; i < graph->num_nodes; ++i) {
        const nano_ir_node_t *n = &graph->nodes[i];
        if (n->opcode == NANO_OP_TERNARY_LINEAR) {
            if (!n->is_ternary) {
                snprintf(err_buf, err_buf_len, "Node %d (%s) marked as linear but is_ternary is false", i, n->name);
                return false;
            }
        } else if (n->opcode == NANO_OP_SSM_STEP) {
            if (n->dim_in != graph->hidden_dim) {
                snprintf(err_buf, err_buf_len, "Node %d (%s) SSM dimension mismatch: expected %d, got %d",
                         i, n->name, graph->hidden_dim, n->dim_in);
                return false;
            }
        }
    }
    return true;
}

void nano_ir_dump_text(const nano_ir_graph_t *graph) {
    if (!graph) return;
    printf("\n=== nano-ir Computational Graph Representation ===\n");
    printf("Model Config: HiddenDim=%d, StateDim=%d, Layers=%d (Nodes=%d)\n",
           graph->hidden_dim, graph->state_dim, graph->num_layers, graph->num_nodes);
    printf("------------------------------------------------------------------\n");
    for (int i = 0; i < graph->num_nodes; ++i) {
        const nano_ir_node_t *n = &graph->nodes[i];
        const char *op_str = "UNKNOWN";
        switch (n->opcode) {
            case NANO_OP_EMBEDDING_LOOKUP: op_str = "nano.embedding_lookup"; break;
            case NANO_OP_RMSNORM:          op_str = "nano.rmsnorm"; break;
            case NANO_OP_TERNARY_LINEAR:   op_str = "nano.ternary_linear [ZERO-MUL]"; break;
            case NANO_OP_SSM_STEP:         op_str = "nano.ssm_step [O(1) MEMORY]"; break;
            case NANO_OP_SILU_MUL:         op_str = "nano.swiglu_gate"; break;
            case NANO_OP_RESIDUAL_ADD:     op_str = "nano.residual_add"; break;
        }
        printf("  %%%d = %-30s in=(%%%d, %%%d) dim=[%d -> %d] scale=%.4f\n",
               n->output_id, op_str, n->input_ids[0], n->input_ids[1],
               n->dim_in, n->dim_out, n->scale);
    }
    printf("==================================================================\n");
}
