/*
 * nano_driver.c — NanoLang front-end orchestration.
 *
 * Drives one translation unit through the pipeline:
 *   source file -> lexer -> parser -> semantic analysis -> in-memory program
 * Backend serialisation lives in nano_codegen.c.
 */

#include "nano_lang.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void nano_options_default(nano_options_t *o) {
    if (!o) return;
    memset(o, 0, sizeof(*o));
    /* 512 MB covers the shipped examples by a wide margin while still catching
     * a typo-sized shape such as hidden = 100000 before it exhausts memory. */
    o->max_total_bytes = 512ull * 1024ull * 1024ull;
}

char *nano_read_source(const char *path, size_t *len_out, nano_diag_t *diag) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        nano_diag_add(diag, 0, "cannot open '%s' for reading", path);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        nano_diag_add(diag, 0, "cannot seek in '%s'", path);
        return NULL;
    }
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        nano_diag_add(diag, 0, "cannot determine the size of '%s'", path);
        return NULL;
    }
    rewind(f);

    char *buf = (char *)malloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        nano_diag_add(diag, 0, "out of memory reading '%s' (%ld bytes)", path, size);
        return NULL;
    }

    size_t got = fread(buf, 1, (size_t)size, f);
    fclose(f);

    if (got != (size_t)size) {
        free(buf);
        nano_diag_add(diag, 0, "short read on '%s' (got %zu of %ld bytes)", path,
                      got, size);
        return NULL;
    }

    buf[size] = '\0';
    if (len_out) *len_out = (size_t)size;
    return buf;
}

int nano_compile_file(const char *path, const nano_options_t *opt,
                      nano_program_t *out, nano_diag_t *diag) {
    if (!path || !out) return -1;
    memset(out, 0, sizeof(*out));
    snprintf(out->source_path, sizeof(out->source_path), "%s", path);

    nano_options_t local;
    if (!opt) {
        nano_options_default(&local);
        opt = &local;
    }

    size_t len = 0;
    char *src = nano_read_source(path, &len, diag);
    if (!src) return -1;

    nano_token_t *tokens = NULL;
    int n_tokens = 0;
    if (nano_lex(src, len, &tokens, &n_tokens, diag) != 0) {
        free(src);
        return -1;
    }
    free(src);

    nano_ast_t *ast = NULL;
    if (nano_parse(tokens, n_tokens, &ast, diag) != 0) {
        free(tokens);
        return -1;
    }
    free(tokens);

    int rc = nano_sema_run(ast, opt, out, diag);
    nano_ast_free(ast);

    if (rc != 0) {
        nano_program_free(out);
        return -1;
    }
    return 0;
}
