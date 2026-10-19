/*
 * nano_codegen.c — NanoLang back end.
 *
 * Serialises the compiled program by calling phase1's nanow_write_file(), so
 * the compiler and the runtime share exactly one container writer. Nothing in
 * this file re-implements the .nanow layout.
 */

#include "nano_lang.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* phase1/src/nanow_writer.c */
extern int nanow_write_file(const char *filepath,
                            const nanow_header_t *hdr_in,
                            const nanow_tensor_desc_t *descs,
                            const void **tensor_payloads);

int nano_emit_nanow(const nano_program_t *p, const char *out_path,
                    nano_diag_t *diag) {
    if (!p || !out_path || !out_path[0]) {
        nano_diag_add(diag, 0, "no output path was given");
        return -1;
    }
    if (p->n_tensors <= 0) {
        nano_diag_add(diag, 0, "refusing to write a container with no tensors");
        return -1;
    }

    nanow_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.flags        = 0;   /* no optional container features are in use yet */
    hdr.num_tensors  = (uint32_t)p->n_tensors;
    hdr.num_layers   = p->num_layers;
    hdr.hidden_dim   = p->hidden_dim;
    hdr.state_dim    = p->state_dim;
    hdr.vocab_size   = p->vocab_size;
    hdr.total_params = p->total_params;
    /* magic, version, file_size and header_size are filled by nanow_write_file */

    nanow_tensor_desc_t *descs = (nanow_tensor_desc_t *)calloc(
        (size_t)p->n_tensors, sizeof(nanow_tensor_desc_t));
    const void **payloads = (const void **)calloc(
        (size_t)p->n_tensors, sizeof(void *));
    if (!descs || !payloads) {
        free(descs);
        free(payloads);
        nano_diag_add(diag, 0, "out of memory preparing the container directory");
        return -1;
    }

    for (int i = 0; i < p->n_tensors; ++i) {
        const nano_tensor_t *t = &p->tensors[i];
        snprintf(descs[i].name, NANOW_MAX_NAME_LEN, "%s", t->name);
        descs[i].dtype        = t->dtype;
        descs[i].packing      = t->packing;
        descs[i].ndims        = t->ndims;
        for (uint32_t d = 0; d < 4; ++d) descs[i].shape[d] = t->shape[d];
        descs[i].num_elements = t->num_elements;
        descs[i].data_size    = t->data_size;
        descs[i].scale_factor = t->scale;
        /* data_offset is computed and validated by nanow_write_file */
        payloads[i] = t->payload;
    }

    int rc = nanow_write_file(out_path, &hdr, descs, payloads);
    free(descs);
    free(payloads);

    if (rc != 0) {
        nano_diag_add(diag, 0, "could not write '%s': %s", out_path,
                      strerror(errno));
        return -1;
    }
    return 0;
}

int nano_write_ir_text(const nano_program_t *p, const char *path,
                       nano_diag_t *diag) {
    if (!p || !path) return -1;
    if (!p->ir_enabled) {
        nano_diag_add(diag, 0,
                      "no op statements were declared, so there is no IR graph to write");
        return -1;
    }

    /* nano_ir_dump_text() from phase1 prints to stdout. Redirect stdout for the
     * duration of the call so the existing formatter is reused instead of being
     * duplicated here. */
    fflush(stdout);
    int saved = dup(fileno(stdout));
    if (saved < 0) {
        nano_diag_add(diag, 0, "could not duplicate stdout for the IR dump");
        return -1;
    }
    if (!freopen(path, "w", stdout)) {
        dup2(saved, fileno(stdout));
        close(saved);
        nano_diag_add(diag, 0, "could not open '%s' for writing", path);
        return -1;
    }

    nano_ir_dump_text(&p->ir);
    fflush(stdout);

    dup2(saved, fileno(stdout));
    close(saved);
    return 0;
}

void nano_program_free(nano_program_t *p) {
    if (!p) return;
    for (int i = 0; i < p->n_tensors; ++i) {
        free(p->tensors[i].payload);
        p->tensors[i].payload = NULL;
    }
    free(p->tensors);
    p->tensors = NULL;
    p->n_tensors = 0;

    if (p->ir_enabled) {
        nano_ir_graph_free(&p->ir);
        p->ir_enabled = 0;
    }
}

static const char *dtype_name(uint32_t d) {
    switch (d) {
        case NANOW_DTYPE_TERNARY: return "ternary";
        case NANOW_DTYPE_INT8:    return "int8";
        case NANOW_DTYPE_FP16:    return "fp16";
        case NANOW_DTYPE_FP32:    return "fp32";
    }
    return "?";
}

static const char *packing_name(uint32_t p) {
    switch (p) {
        case NANOW_PACKING_RAW_INT8: return "raw";
        case NANOW_PACKING_2BIT:     return "2bit";
        case NANOW_PACKING_RADIX3:   return "radix3";
    }
    return "?";
}

void nano_print_manifest(const nano_program_t *p, FILE *out) {
    if (!p || !out) return;

    fprintf(out, "model: layers=%u hidden=%u state=%u vocab=%u\n",
            p->num_layers, p->hidden_dim, p->state_dim, p->vocab_size);
    fprintf(out, "tensors: %d   total_params: %llu\n", p->n_tensors,
            (unsigned long long)p->total_params);
    fprintf(out, "%-40s %-8s %-6s %-18s %10s %10s\n",
            "name", "dtype", "pack", "shape", "elements", "bytes");

    uint64_t payload_total = 0;
    for (int i = 0; i < p->n_tensors; ++i) {
        const nano_tensor_t *t = &p->tensors[i];
        char shape[64] = {0};
        size_t off = 0;
        for (uint32_t d = 0; d < t->ndims; ++d) {
            off += (size_t)snprintf(shape + off, sizeof(shape) - off,
                                    d ? "x%u" : "%u", t->shape[d]);
        }
        fprintf(out, "%-40s %-8s %-6s %-18s %10llu %10zu\n", t->name,
                dtype_name(t->dtype), packing_name(t->packing), shape,
                (unsigned long long)t->num_elements, t->data_size);
        payload_total += t->data_size;
    }
    fprintf(out, "payload total: %llu bytes (before page alignment)\n",
            (unsigned long long)payload_total);
}
