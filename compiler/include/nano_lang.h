/*
 * nano_lang.h — NanoLang: a declarative model description language.
 *
 * NanoLang source (`.nano`) compiles with the `nanoc` compiler to a `.nanow`
 * binary container that the existing C loader (phase1/include/nanow_mmap.h) and
 * the Python runner (run_smollm2.load_nanow) read without modification.
 *
 * This header is the language's public surface: tokens, AST, diagnostics and the
 * compile/emit API. The unit tests inspect the AST directly, so it is exposed on
 * purpose rather than kept private to the parser.
 *
 * Spec and grammar: compiler/README.md
 */

#ifndef NANO_LANG_H
#define NANO_LANG_H

#include "nanow_format.h"
#include "nano_ir.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Diagnostics                                                               */
/* ------------------------------------------------------------------------- */

#define NANO_MAX_DIAGS 32
#define NANO_DIAG_MSG  256

typedef struct {
    int  line;
    int  severity;                /* NANO_SEV_ERROR or NANO_SEV_WARNING */
    char msg[NANO_DIAG_MSG];
} nano_diag_item_t;

#define NANO_SEV_ERROR   0
#define NANO_SEV_WARNING 1

typedef struct {
    int count;
    int overflow;                 /* further diagnostics were dropped */
    nano_diag_item_t items[NANO_MAX_DIAGS];
} nano_diag_t;

void nano_diag_init(nano_diag_t *d);
void nano_diag_add(nano_diag_t *d, int line, const char *fmt, ...);
void nano_diag_warn(nano_diag_t *d, int line, const char *fmt, ...);
/* Number of diagnostics with NANO_SEV_ERROR. Warnings do not count. */
int  nano_diag_error_count(const nano_diag_t *d);
/* Prints "path:line: error|warning: msg" lines, then a trailing summary. */
void nano_diag_print(const nano_diag_t *d, const char *path, FILE *out);

/* ------------------------------------------------------------------------- */
/* Lexer                                                                     */
/* ------------------------------------------------------------------------- */

typedef enum {
    NANO_TOK_EOF = 0,
    NANO_TOK_IDENT,
    NANO_TOK_NUMBER,
    NANO_TOK_STRING,
    NANO_TOK_LBRACE,      /* {   */
    NANO_TOK_RBRACE,      /* }   */
    NANO_TOK_LBRACKET,    /* [   */
    NANO_TOK_RBRACKET,    /* ]   */
    NANO_TOK_COMMA,       /* ,   */
    NANO_TOK_EQUALS,      /* =   */
    NANO_TOK_DOTDOT       /* ..  */
} nano_token_kind_t;

typedef struct {
    nano_token_kind_t kind;
    int    line;
    char   text[192];     /* ident text / decoded string / number literal */
    double number;
    int    is_float;
} nano_token_t;

const char *nano_token_kind_name(nano_token_kind_t k);

/* Lexes `src` (NUL-terminated or `len`-long). On success *out holds a
 * malloc'd array of *count tokens ending with NANO_TOK_EOF; the caller frees
 * it with free(). Returns 0 on success, -1 if diagnostics were produced. */
int nano_lex(const char *src, size_t len, nano_token_t **out, int *count,
             nano_diag_t *diag);

/* ------------------------------------------------------------------------- */
/* AST                                                                       */
/* ------------------------------------------------------------------------- */

typedef enum {
    NANO_VAL_INT = 0,
    NANO_VAL_FLOAT,
    NANO_VAL_STRING,
    NANO_VAL_IDENT,
    NANO_VAL_SHAPE
} nano_val_kind_t;

typedef struct {
    char            key[64];
    nano_val_kind_t kind;
    double          num;
    char            str[192];
    int             shape[4];
    int             n_shape;
    int             line;
} nano_field_t;

typedef enum {
    NANO_ST_MODEL = 0,
    NANO_ST_TENSOR,
    NANO_ST_OP,
    NANO_ST_FOR,
    NANO_ST_EMIT
} nano_stmt_kind_t;

typedef struct nano_stmt {
    nano_stmt_kind_t kind;
    int              line;

    char name[192];       /* model name / tensor name (raw, may hold ${..})   */
    char opcode[32];      /* NANO_ST_OP: opcode identifier                    */
    char var[32];         /* NANO_ST_FOR: loop variable                       */
    int  lo, hi;          /* NANO_ST_FOR: half-open range [lo, hi)            */

    nano_field_t *fields;
    int           n_fields;

    struct nano_stmt *body;   /* NANO_ST_FOR */
    int               n_body;
} nano_stmt_t;

typedef struct nano_ast {
    nano_stmt_t *stmts;
    int          n_stmts;
} nano_ast_t;

const char *nano_stmt_kind_name(nano_stmt_kind_t k);

/* Parses a token stream. Returns 0 on success. */
int  nano_parse(const nano_token_t *tokens, int count, nano_ast_t **out,
                nano_diag_t *diag);
void nano_ast_free(nano_ast_t *ast);

/* ------------------------------------------------------------------------- */
/* Compiled program                                                          */
/* ------------------------------------------------------------------------- */

typedef struct {
    char     name[NANOW_MAX_NAME_LEN];
    uint32_t dtype;                /* nanow_dtype_t        */
    uint32_t packing;              /* nanow_packing_t      */
    uint32_t ndims;
    uint32_t shape[4];
    uint64_t num_elements;
    size_t   data_size;            /* payload bytes on disk */
    float    scale;
    uint8_t *payload;
    int      line;
} nano_tensor_t;

typedef struct {
    char     source_path[256];

    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t state_dim;
    uint32_t vocab_size;
    uint64_t total_params;

    int            n_tensors;
    nano_tensor_t *tensors;

    /* Compile-time graph. Validated during compilation and available for
     * inspection; the runtime reads the container, not this graph. */
    nano_ir_graph_t ir;
    int             ir_enabled;

    /* Output path from an `emit` statement; the CLI's -o overrides it. */
    char emit_path[256];
    int  have_emit;
} nano_program_t;

typedef struct {
    int         dump_ir;           /* print the IR graph to stdout            */
    const char *ir_out;            /* also write a .nanoir text file          */
    int         print_manifest;    /* print the compiled tensor table         */
    const char *from_dir;          /* base dir for `from "..."` (NULL = src)  */
    uint64_t    max_total_bytes;   /* allocation guard over all payloads      */
    /* Off by default. A tensor the runtime cannot fully read is an error, so
     * that the compiler never silently emits a partly-loadable container. */
    int         allow_unreadable;
} nano_options_t;

void nano_options_default(nano_options_t *o);

/* Full pipeline: read file -> lex -> parse -> sema -> codegen in memory.
 * Returns 0 on success; on failure `diag` holds the reasons and *out is
 * zeroed. `out` must be released with nano_program_free(). */
int  nano_compile_file(const char *path, const nano_options_t *opt,
                       nano_program_t *out, nano_diag_t *diag);

void nano_program_free(nano_program_t *p);

/* Serialises via phase1's nanow_write_file(). Returns 0 on success. */
int  nano_emit_nanow(const nano_program_t *p, const char *out_path,
                     nano_diag_t *diag);

/* Writes the IR graph as text (the format nano_ir_dump_text() prints). */
int  nano_write_ir_text(const nano_program_t *p, const char *path,
                        nano_diag_t *diag);

/* Prints the compiled tensor table to `out`. */
void nano_print_manifest(const nano_program_t *p, FILE *out);

/* Payload size in bytes for a dtype/packing/element count, or 0 if invalid. */
size_t nano_payload_bytes(uint32_t dtype, uint32_t packing, uint64_t num_elements);

/* ------------------------------------------------------------------------- */
/* Pipeline stages (exposed so the unit tests can drive them separately)     */
/* ------------------------------------------------------------------------- */

/* Semantic analysis: unrolls loops, expands ${var}, validates fields, builds
 * the tensor table and lowers `op` statements into the nano-ir graph.
 * `prog` must be zeroed by the caller. Returns 0 on success. */
int nano_sema_run(const nano_ast_t *ast, const nano_options_t *opt,
                  nano_program_t *prog, nano_diag_t *diag);

/* Reads a whole file into a malloc'd, NUL-terminated buffer. Returns NULL and
 * records a diagnostic on failure. */
char *nano_read_source(const char *path, size_t *len_out, nano_diag_t *diag);

#ifdef __cplusplus
}
#endif

#endif /* NANO_LANG_H */
