/*
 * test_nanoc.c — unit and integration tests for the NanoLang compiler.
 *
 * Run with `make -C compiler test`, or build and invoke build/test_nanoc.
 *
 * Two kinds of check are mixed on purpose:
 *   - unit tests drive the lexer, parser and sema over source text
 *   - the round-trip test compiles a .nano, writes a .nanow, and then opens it
 *     with phase1's real reader (nanow_model_open), so a change to the
 *     container layout that the compiler gets wrong fails here rather than at
 *     runtime.
 *
 * The ternary decoder at the bottom is written from the format description
 * rather than calling back into the packer, so the packing is checked against
 * an independent implementation instead of against itself.
 */

#include "nano_lang.h"
#include "nanow_mmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* phase1/src/nanow_writer.c — the one container writer shared by the compiler,
 * the Python exporter and the cache. Declared here so a test can drive its
 * header flags directly instead of going through the .nano front end. */
extern int nanow_write_file(const char *filepath,
                            const nanow_header_t *hdr_in,
                            const nanow_tensor_desc_t *descs,
                            const void **tensor_payloads);

static int g_pass = 0;
static int g_fail = 0;
static const char *g_current = "";
static char g_tmpdir[256];

#define RUN(fn) do { g_current = #fn; fn(); } while (0)

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        if (cond) {                                                             \
            g_pass++;                                                           \
        } else {                                                                \
            g_fail++;                                                           \
            fprintf(stderr, "  FAIL %s:%d [%s]: ", __func__, __LINE__,          \
                    g_current);                                                 \
            fprintf(stderr, __VA_ARGS__);                                       \
            fprintf(stderr, "\n");                                              \
        }                                                                       \
    } while (0)

/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */

static const char *tpath(const char *name, char *out, size_t cap) {
    snprintf(out, cap, "%s/%s", g_tmpdir, name);
    return out;
}

static int write_text(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t n = strlen(text);
    int ok = (fwrite(text, 1, n, f) == n);
    fclose(f);
    return ok ? 0 : -1;
}

static int compile_src(const char *src, const char *fname, nano_program_t *prog,
                       nano_diag_t *diag, nano_options_t *opt) {
    char path[512];
    tpath(fname, path, sizeof(path));
    if (write_text(path, src) != 0) return -99;

    nano_diag_init(diag);
    nano_options_t local;
    if (!opt) {
        nano_options_default(&local);
        opt = &local;
    }
    memset(prog, 0, sizeof(*prog));
    return nano_compile_file(path, opt, prog, diag);
}

static int diag_has(const nano_diag_t *d, const char *needle) {
    for (int i = 0; i < d->count; ++i) {
        if (strstr(d->items[i].msg, needle)) return 1;
    }
    return 0;
}

static int diag_count_severity(const nano_diag_t *d, int severity) {
    int n = 0;
    for (int i = 0; i < d->count; ++i) {
        if (d->items[i].severity == severity) n++;
    }
    return n;
}

/* A source that is valid unless a test says otherwise. */
static const char *kMinimal =
    "model m {\n"
    "  layers = 1\n"
    "  hidden = 4\n"
    "  vocab  = 8\n"
    "}\n"
    "tensor w {\n"
    "  dtype = fp32\n"
    "  shape = [4]\n"
    "  init  = ones\n"
    "}\n";

/* ------------------------------------------------------------------------- */
/* Lexer                                                                     */
/* ------------------------------------------------------------------------- */

static void lexer_basic_kinds(void) {
    const char *src = "model m { layers = 1 }";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) == 0, "lex should succeed");
    CHECK(n == 8, "expected 8 tokens, got %d", n);
    CHECK(toks[0].kind == NANO_TOK_IDENT && strcmp(toks[0].text, "model") == 0,
          "token 0 should be ident 'model'");
    CHECK(toks[2].kind == NANO_TOK_LBRACE, "token 2 should be '{'");
    CHECK(toks[4].kind == NANO_TOK_EQUALS, "token 4 should be '='");
    CHECK(toks[5].kind == NANO_TOK_NUMBER && toks[5].number == 1.0,
          "token 5 should be the number 1");
    CHECK(toks[7].kind == NANO_TOK_EOF, "last token should be EOF");
    free(toks);
}

static void lexer_range_operator(void) {
    const char *src = "for i in 0..4";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) == 0, "lex should succeed");
    CHECK(n == 7, "expected 7 tokens, got %d", n);
    CHECK(toks[3].kind == NANO_TOK_NUMBER && toks[3].number == 0.0,
          "token 3 should be the number 0");
    CHECK(toks[4].kind == NANO_TOK_DOTDOT, "token 4 should be '..'");
    CHECK(toks[5].number == 4.0, "token 5 should be the number 4");
    CHECK(toks[3].is_float == 0, "'0' must not be lexed as a float");
    free(toks);
}

static void lexer_float_literal(void) {
    const char *src = "scale = 0.02";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) == 0, "lex should succeed");
    const nano_token_t *num = NULL;
    for (int i = 0; i < n; ++i) {
        if (toks[i].kind == NANO_TOK_NUMBER) num = &toks[i];
    }
    CHECK(num && num->is_float == 1, "'0.02' should be a float token");
    CHECK(num && num->number > 0.0199 && num->number < 0.0201,
          "'0.02' should parse to about 0.02");
    free(toks);
}

static void lexer_comments_are_skipped(void) {
    const char *src =
        "// leading comment\n"
        "model /* inline */ m {\n"
        "  /* multi\n"
        "     line */ layers = 1\n"
        "}\n";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) == 0, "lex should succeed");
    CHECK(toks[0].kind == NANO_TOK_IDENT && strcmp(toks[0].text, "model") == 0,
          "the first token should be 'model', not a comment fragment");
    CHECK(toks[0].line == 2, "'model' should be on line 2, got %d", toks[0].line);
    /* The block comment spans a newline, so 'layers' is on line 4. */
    int layers_line = -1;
    for (int i = 0; i < n; ++i) {
        if (toks[i].kind == NANO_TOK_IDENT && strcmp(toks[i].text, "layers") == 0) {
            layers_line = toks[i].line;
        }
    }
    CHECK(layers_line == 4, "'layers' should be on line 4, got %d", layers_line);
    free(toks);
}

static void lexer_string_keeps_interpolation(void) {
    const char *src = "\"a${b}c\"";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) == 0, "lex should succeed");
    CHECK(n == 2, "expected a string and EOF, got %d tokens", n);
    CHECK(toks[0].kind == NANO_TOK_STRING, "token 0 should be a string");
    CHECK(strcmp(toks[0].text, "a${b}c") == 0,
          "interpolation must survive lexing verbatim, got '%s'", toks[0].text);
    free(toks);
}

static void lexer_string_escapes(void) {
    const char *src = "\"a\\\"b\\\\c\"";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) == 0, "lex should succeed");
    CHECK(strcmp(toks[0].text, "a\"b\\c") == 0,
          "escapes should decode to a\"b\\c, got '%s'", toks[0].text);
    free(toks);
}

static void lexer_unterminated_string_fails(void) {
    const char *src = "tensor \"oops {\n dtype = fp32\n}\n";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) != 0, "lex should fail");
    CHECK(diag_has(&d, "unterminated string"), "should report an unterminated string");
    free(toks);
}

static void lexer_unexpected_character_fails(void) {
    const char *src = "model @ m {}\n";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) != 0, "lex should fail");
    CHECK(diag_has(&d, "unexpected character"), "should report the bad character");
    free(toks);
}

static void lexer_unterminated_block_comment_fails(void) {
    const char *src = "model m { /* never closed\n";
    nano_token_t *toks = NULL;
    int n = 0;
    nano_diag_t d;
    nano_diag_init(&d);

    CHECK(nano_lex(src, strlen(src), &toks, &n, &d) != 0, "lex should fail");
    CHECK(diag_has(&d, "unterminated block comment"),
          "should report the unterminated comment");
    free(toks);
}

/* ------------------------------------------------------------------------- */
/* Parser                                                                    */
/* ------------------------------------------------------------------------- */

/* Parses `src` into a fresh AST; returns NULL on failure. */
static nano_ast_t *parse_only(const char *src, nano_diag_t *diag) {
    nano_token_t *toks = NULL;
    int n = 0;
    if (nano_lex(src, strlen(src), &toks, &n, diag) != 0) {
        free(toks);
        return NULL;
    }
    nano_ast_t *ast = NULL;
    int rc = nano_parse(toks, n, &ast, diag);
    free(toks);
    return rc == 0 ? ast : NULL;
}

static void parser_model_fields(void) {
    const char *src = "model m {\n layers = 2\n hidden = 16\n}\n";
    nano_diag_t d;
    nano_diag_init(&d);
    nano_ast_t *ast = parse_only(src, &d);

    CHECK(ast != NULL, "parse should succeed");
    if (!ast) return;
    CHECK(ast->n_stmts == 1, "expected 1 statement, got %d", ast->n_stmts);
    const nano_stmt_t *st = &ast->stmts[0];
    CHECK(st->kind == NANO_ST_MODEL, "statement should be a model block");
    CHECK(strcmp(st->name, "m") == 0, "model name should be 'm'");
    CHECK(st->n_fields == 2, "expected 2 fields, got %d", st->n_fields);
    CHECK(strcmp(st->fields[0].key, "layers") == 0, "first field should be 'layers'");
    CHECK(st->fields[0].kind == NANO_VAL_INT && st->fields[0].num == 2.0,
          "'layers' should be the int 2");
    nano_ast_free(ast);
}

static void parser_shape_value(void) {
    const char *src =
        "model m { layers = 1 hidden = 1 }\n"
        "tensor t { dtype = fp32 shape = [2, 3, 4] init = ones }\n";
    nano_diag_t d;
    nano_diag_init(&d);
    nano_ast_t *ast = parse_only(src, &d);

    CHECK(ast != NULL, "parse should succeed");
    if (!ast || ast->n_stmts < 2) { nano_ast_free(ast); return; }
    const nano_field_t *f = NULL;
    for (int i = 0; i < ast->stmts[1].n_fields; ++i) {
        if (strcmp(ast->stmts[1].fields[i].key, "shape") == 0) f = &ast->stmts[1].fields[i];
    }
    CHECK(f && f->kind == NANO_VAL_SHAPE, "'shape' should be a shape value");
    CHECK(f && f->n_shape == 3, "expected 3 dimensions, got %d", f ? f->n_shape : -1);
    CHECK(f && f->shape[0] == 2 && f->shape[1] == 3 && f->shape[2] == 4,
          "dimensions should be 2, 3, 4");
    nano_ast_free(ast);
}

static void parser_for_body_and_ident_name(void) {
    const char *src =
        "model m { layers = 1 hidden = 1 }\n"
        "for i in 0..2 {\n"
        "  tensor t { dtype = fp32 shape = [1] init = ones }\n"
        "}\n";
    nano_diag_t d;
    nano_diag_init(&d);
    nano_ast_t *ast = parse_only(src, &d);

    CHECK(ast != NULL, "parse should succeed");
    if (!ast || ast->n_stmts < 2) { nano_ast_free(ast); return; }
    const nano_stmt_t *loop = &ast->stmts[1];
    CHECK(loop->kind == NANO_ST_FOR, "statement should be a for block");
    CHECK(strcmp(loop->var, "i") == 0, "loop variable should be 'i'");
    CHECK(loop->lo == 0 && loop->hi == 2, "range should be 0..2");
    CHECK(loop->n_body == 1, "loop body should hold 1 statement, got %d", loop->n_body);
    CHECK(loop->n_body == 1 && strcmp(loop->body[0].name, "t") == 0,
          "the loop body statement should be tensor 't'");
    nano_ast_free(ast);
}

static void parser_for_closes_its_brace(void) {
    /* Regression: the for-body '}' used to be left to the enclosing block,
     * which produced a spurious "unexpected '}'" at the next top-level line. */
    const char *src =
        "model m { layers = 1 hidden = 1 }\n"
        "for i in 0..1 {\n"
        "  tensor a { dtype = fp32 shape = [1] init = ones }\n"
        "}\n"
        "tensor b { dtype = fp32 shape = [1] init = ones }\n";
    nano_diag_t d;
    nano_diag_init(&d);
    nano_ast_t *ast = parse_only(src, &d);

    CHECK(ast != NULL, "parse should succeed");
    CHECK(ast && ast->n_stmts == 3, "expected model, for and tensor, got %d",
          ast ? ast->n_stmts : -1);
    nano_ast_free(ast);
}

static void parser_missing_equals_fails(void) {
    const char *src = "model m { layers 1 }\n";
    nano_diag_t d;
    nano_diag_init(&d);
    nano_ast_t *ast = parse_only(src, &d);

    CHECK(ast == NULL, "parse should fail");
    CHECK(diag_has(&d, "expected '='"), "should ask for '=' after the field name");
    nano_ast_free(ast);
}

static void parser_stray_brace_fails(void) {
    const char *src = "}\n";
    nano_diag_t d;
    nano_diag_init(&d);
    nano_ast_t *ast = parse_only(src, &d);

    CHECK(ast == NULL, "parse should fail");
    CHECK(diag_has(&d, "expected a statement"), "should ask for a statement");
    nano_ast_free(ast);
}

static void parser_non_integer_loop_range_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 1 }\n"
        "for i in 0.5..2 {\n}\n";
    nano_diag_t d;
    nano_diag_init(&d);
    nano_ast_t *ast = parse_only(src, &d);

    CHECK(ast == NULL, "parse should fail");
    CHECK(diag_has(&d, "must be an integer"), "should reject a fractional start");
    nano_ast_free(ast);
}

/* ------------------------------------------------------------------------- */
/* Semantic analysis                                                         */
/* ------------------------------------------------------------------------- */

static void sema_minimal_ok(void) {
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(kMinimal, "min.nano", &p, &d, NULL) == 0,
          "the minimal source should compile");
    CHECK(p.n_tensors == 1, "expected 1 tensor, got %d", p.n_tensors);
    CHECK(p.total_params == 4, "expected 4 params, got %llu",
          (unsigned long long)p.total_params);
    CHECK(p.num_layers == 1 && p.hidden_dim == 4 && p.vocab_size == 8,
          "architecture fields should be carried through");
    CHECK(p.tensors[0].data_size == 16, "fp32[4] should be 16 bytes, got %zu",
          p.tensors[0].data_size);
    CHECK(p.tensors[0].payload != NULL, "payload should be allocated");
    nano_program_free(&p);
}

static void sema_payload_size_table(void) {
    CHECK(nano_payload_bytes(NANOW_DTYPE_TERNARY, NANOW_PACKING_2BIT, 4) == 1,
          "4 ternary values should be 1 byte");
    CHECK(nano_payload_bytes(NANOW_DTYPE_TERNARY, NANOW_PACKING_2BIT, 10) == 3,
          "10 ternary values should round up to 3 bytes");
    CHECK(nano_payload_bytes(NANOW_DTYPE_TERNARY, NANOW_PACKING_RADIX3, 5) == 1,
          "5 radix3 values should be 1 byte");
    CHECK(nano_payload_bytes(NANOW_DTYPE_INT8, NANOW_PACKING_RAW_INT8, 7) == 7,
          "7 int8 values should be 7 bytes");
    CHECK(nano_payload_bytes(NANOW_DTYPE_FP16, NANOW_PACKING_RAW_INT8, 7) == 14,
          "7 fp16 values should be 14 bytes");
    CHECK(nano_payload_bytes(NANOW_DTYPE_FP32, NANOW_PACKING_RAW_INT8, 7) == 28,
          "7 fp32 values should be 28 bytes");
}

static void sema_duplicate_tensor_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = ones }\n"
        "tensor w { dtype = fp32 shape = [4] init = zeros }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "dup.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "duplicate tensor name"), "should report the duplicate");
    nano_program_free(&p);
}

static void sema_ternary_requires_scale(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = ternary shape = [4, 4] init = ternary }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "noscale.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "needs an explicit 'scale'"), "should ask for a scale");
    nano_program_free(&p);
}

static void sema_ternary_scale_must_be_positive(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = ternary shape = [4, 4] scale = 0.0 init = ternary }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "zscale.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "must be greater than zero"), "should reject scale 0");
    nano_program_free(&p);
}

static void sema_unbound_variable_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor \"a.${nope}\" { dtype = fp32 shape = [4] init = ones }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "unbound.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "unbound loop variable '${nope}'"),
          "should name the unbound variable");
    nano_program_free(&p);
}

static void sema_loop_unrolls_and_interpolates(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "for i in 0..3 {\n"
        "  tensor \"a.${i}\" { dtype = fp32 shape = [4] init = ones }\n"
        "}\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "unroll.nano", &p, &d, NULL) == 0, "should compile");
    CHECK(p.n_tensors == 3, "expected 3 tensors, got %d", p.n_tensors);
    CHECK(p.n_tensors == 3 && strcmp(p.tensors[0].name, "a.0") == 0,
          "first name should be a.0, got '%s'", p.n_tensors ? p.tensors[0].name : "-");
    CHECK(p.n_tensors == 3 && strcmp(p.tensors[2].name, "a.2") == 0,
          "last name should be a.2");
    nano_program_free(&p);
}

static void sema_unknown_field_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = ones bogus = 1 }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "unknown.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "unknown field 'bogus'"), "should name the unknown field");
    nano_program_free(&p);
}

static void sema_duplicate_field_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 dtype = fp32 shape = [4] init = ones }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "dupfield.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "duplicate field 'dtype'"), "should report the duplicate field");
    nano_program_free(&p);
}

static void sema_missing_model_fails(void) {
    const char *src = "tensor w { dtype = fp32 shape = [4] init = ones }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "nomodel.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "no model block"), "should ask for a model block");
    nano_program_free(&p);
}

static void sema_no_tensors_fails(void) {
    const char *src = "model m { layers = 1 hidden = 4 }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "notensors.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "no tensors"), "should report that nothing would be written");
    nano_program_free(&p);
}

static void sema_zero_dimension_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [0] init = ones }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "zerodim.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "greater than zero"), "should reject a zero dimension");
    nano_program_free(&p);
}

static void sema_long_tensor_name_fails(void) {
    /* 64 bytes is the descriptor's hard limit (NANOW_MAX_NAME_LEN). */
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" {\n"
        "  dtype = fp32 shape = [4] init = ones\n"
        "}\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "longname.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "the container stores at most"),
          "should explain the container's name limit");
    nano_program_free(&p);
}

static void sema_max_bytes_guard(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [1024] init = ones }\n";
    nano_program_t p;
    nano_diag_t d;
    nano_options_t opt;
    nano_options_default(&opt);
    opt.max_total_bytes = 64;
    CHECK(compile_src(src, "toobig.nano", &p, &d, &opt) != 0, "should fail");
    CHECK(diag_has(&d, "over the limit"), "should explain the limit was exceeded");
    nano_program_free(&p);
}

static void sema_emit_statement_recorded(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = ones }\n"
        "emit \"out.nanow\"\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "emit.nano", &p, &d, NULL) == 0, "should compile");
    CHECK(p.have_emit == 1, "emit should be recorded");
    CHECK(strcmp(p.emit_path, "out.nanow") == 0, "emit path should be 'out.nanow'");
    nano_program_free(&p);
}

/* ------------------------------------------------------------------------- */
/* IR lowering                                                               */
/* ------------------------------------------------------------------------- */

static void sema_ops_lower_into_ir(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 state = 2 }\n"
        "tensor w { dtype = ternary shape = [4, 4] scale = 0.1 init = ternary }\n"
        "op ternary_linear \"lin\" { in = 4 out = 4 weight = \"w\" }\n"
        "op ssm_step \"rec\" { in = 4 out = 4 }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "ir.nano", &p, &d, NULL) == 0, "should compile");
    CHECK(p.ir_enabled == 1, "the IR graph should be enabled");
    CHECK(p.ir.num_nodes == 2, "expected 2 IR nodes, got %d", p.ir.num_nodes);
    CHECK(p.ir.num_nodes == 2 && p.ir.nodes[0].opcode == NANO_OP_TERNARY_LINEAR,
          "node 0 should be a ternary linear");
    CHECK(p.ir.num_nodes == 2 && p.ir.nodes[0].is_ternary,
          "a ternary linear node must be flagged ternary");
    CHECK(p.ir.num_nodes == 2 && p.ir.nodes[1].opcode == NANO_OP_SSM_STEP,
          "node 1 should be an ssm step");
    nano_program_free(&p);
}

static void sema_ssm_dim_must_match_hidden(void) {
    /* Caught by phase1's nano_ir_validate_graph, not by our own checks. A
     * tensor is declared so that the tensor-count check stays quiet and the
     * only remaining complaint is the one under test. */
    const char *src =
        "model m { layers = 1 hidden = 4 state = 2 }\n"
        "tensor g { dtype = fp32 shape = [4] init = ones }\n"
        "op ssm_step \"rec\" { in = 8 out = 8 }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "ssmmismatch.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "nano-ir validation failed"),
          "the failure should come from the IR validator");
    nano_program_free(&p);
}

static void sema_ternary_linear_weight_checked(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4, 4] init = ones }\n"
        "op ternary_linear \"lin\" { in = 4 out = 4 weight = \"w\" }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "f32weight.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "needs a ternary weight"), "should reject a non-ternary weight");
    nano_program_free(&p);
}

static void sema_ternary_linear_shape_checked(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = ternary shape = [4, 4] scale = 0.1 init = ternary }\n"
        "op ternary_linear \"lin\" { in = 4 out = 8 weight = \"w\" }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "badshape.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "declares out=8 in=4"), "should show the mismatch");
    nano_program_free(&p);
}

static void sema_unknown_weight_tensor_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "op ternary_linear \"lin\" { in = 4 out = 4 weight = \"ghost\" }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "ghost.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "not a declared tensor"), "should report the missing tensor");
    nano_program_free(&p);
}

static void sema_unknown_opcode_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "op warp_drive \"x\" { in = 4 out = 4 }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "badop.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "unknown opcode 'warp_drive'"), "should name the opcode");
    nano_program_free(&p);
}

/* ------------------------------------------------------------------------- */
/* Runtime-compatibility gates                                               */
/* ------------------------------------------------------------------------- */

static void sema_fp16_is_gated(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp16 shape = [4] init = ones }\n";
    nano_program_t p;
    nano_diag_t d;

    CHECK(compile_src(src, "fp16.nano", &p, &d, NULL) != 0,
          "fp16 should be rejected by default");
    CHECK(diag_has(&d, "skips this tensor with a warning"),
          "should explain which reader cannot decode it");
    nano_program_free(&p);

    nano_options_t opt;
    nano_options_default(&opt);
    opt.allow_unreadable = 1;
    CHECK(compile_src(src, "fp16.nano", &p, &d, &opt) == 0,
          "fp16 should compile with --allow-unreadable");
    CHECK(diag_count_severity(&d, NANO_SEV_WARNING) == 1,
          "the downgrade should be a warning");
    CHECK(diag_count_severity(&d, NANO_SEV_ERROR) == 0,
          "there should be no errors left");
    CHECK(p.tensors[0].data_size == 8, "fp16[4] should be 8 bytes");
    /* 1.0f in IEEE binary16 is 0x3C00. */
    CHECK(p.tensors[0].payload[0] == 0x00 && p.tensors[0].payload[1] == 0x3C,
          "fp16 ones should hold 0x3C00 little-endian");
    nano_program_free(&p);
}

static void sema_radix3_is_gated(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = ternary shape = [5] packing = radix3 scale = 0.1 init = ternary }\n";
    nano_program_t p;
    nano_diag_t d;

    CHECK(compile_src(src, "radix3.nano", &p, &d, NULL) != 0,
          "radix3 should be rejected by default");
    CHECK(diag_has(&d, "no unpacker"), "should explain that nothing can decode it");
    nano_program_free(&p);

    nano_options_t opt;
    nano_options_default(&opt);
    opt.allow_unreadable = 1;
    CHECK(compile_src(src, "radix3.nano", &p, &d, &opt) == 0,
          "radix3 should compile with --allow-unreadable");
    CHECK(p.tensors[0].packing == NANOW_PACKING_RADIX3, "packing should be radix3");
    CHECK(p.tensors[0].data_size == 1, "5 radix3 values should be 1 byte");
    nano_program_free(&p);
}

/* ------------------------------------------------------------------------- */
/* `init = from` and deterministic generation                                */
/* ------------------------------------------------------------------------- */

static void sema_from_reads_file(void) {
    char blob[512];
    tpath("blob.bin", blob, sizeof(blob));

    float values[4] = { 0.5f, -1.5f, 2.25f, 0.0f };
    FILE *f = fopen(blob, "wb");
    CHECK(f != NULL, "should be able to create the source blob");
    if (f) {
        fwrite(values, sizeof(float), 4, f);
        fclose(f);
    }

    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = from source = \"blob.bin\" }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "from.nano", &p, &d, NULL) == 0, "should compile");
    if (p.n_tensors == 1) {
        const float *got = (const float *)p.tensors[0].payload;
        CHECK(got[0] == 0.5f && got[1] == -1.5f && got[2] == 2.25f && got[3] == 0.0f,
              "the payload should match the file exactly");
    }
    nano_program_free(&p);
}

static void sema_from_missing_file_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = from source = \"absent.bin\" }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "frommissing.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "cannot open source file"), "should report the missing file");
    nano_program_free(&p);
}

static void sema_from_wrong_size_fails(void) {
    /* 3 floats where the tensor wants 4: silently padding or truncating here
     * would corrupt a weight matrix in a way that only shows up numerically. */
    char blob[512];
    tpath("short.bin", blob, sizeof(blob));
    FILE *f = fopen(blob, "wb");
    CHECK(f != NULL, "should create the short blob");
    if (f) {
        float values[3] = { 1.0f, 2.0f, 3.0f };
        fwrite(values, sizeof(float), 3, f);
        fclose(f);
    }

    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = from source = \"short.bin\" }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "short.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "is 12 bytes but the tensor needs 16"),
          "should report both sizes");
    nano_program_free(&p);
}

static void sema_source_without_from_fails(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = ones source = \"x.bin\" }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "srcno.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "only valid with init = from"), "should explain the pairing");
    nano_program_free(&p);
}

static void sema_seed_requires_ternary_init(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = ones seed = 3 }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "seed.nano", &p, &d, NULL) != 0, "should fail");
    CHECK(diag_has(&d, "only valid with init = ternary"), "should explain the pairing");
    nano_program_free(&p);
}

static void sema_ternary_generation_is_deterministic(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = ternary shape = [64, 64] scale = 0.1 init = ternary }\n";
    nano_program_t a, b;
    nano_diag_t d;
    CHECK(compile_src(src, "det.nano", &a, &d, NULL) == 0, "first compile");
    CHECK(compile_src(src, "det.nano", &b, &d, NULL) == 0, "second compile");
    CHECK(a.n_tensors == 1 && b.n_tensors == 1, "both should have one tensor");
    if (a.n_tensors == 1 && b.n_tensors == 1) {
        CHECK(a.tensors[0].data_size == b.tensors[0].data_size, "same size");
        CHECK(memcmp(a.tensors[0].payload, b.tensors[0].payload,
                     a.tensors[0].data_size) == 0,
              "recompiling the same source must reproduce the same bytes");
    }
    nano_program_free(&a);
    nano_program_free(&b);
}

static void sema_ternary_seed_follows_the_name(void) {
    /* Two identical declarations with different names must not produce
     * identical weights, otherwise a loop over layers would emit N copies of
     * one matrix. */
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor a { dtype = ternary shape = [64, 64] scale = 0.1 init = ternary }\n"
        "tensor b { dtype = ternary shape = [64, 64] scale = 0.1 init = ternary }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "names.nano", &p, &d, NULL) == 0, "should compile");
    if (p.n_tensors == 2) {
        CHECK(memcmp(p.tensors[0].payload, p.tensors[1].payload,
                     p.tensors[0].data_size) != 0,
              "tensors named 'a' and 'b' should differ");
    }
    nano_program_free(&p);
}

static void sema_explicit_seed_overrides_the_name(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor a { dtype = ternary shape = [64, 64] scale = 0.1 init = ternary seed = 7 }\n"
        "tensor b { dtype = ternary shape = [64, 64] scale = 0.1 init = ternary seed = 7 }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "sameed.nano", &p, &d, NULL) == 0, "should compile");
    if (p.n_tensors == 2) {
        CHECK(memcmp(p.tensors[0].payload, p.tensors[1].payload,
                     p.tensors[0].data_size) == 0,
              "an explicit seed should make the names irrelevant");
    }
    nano_program_free(&p);
}

/* ------------------------------------------------------------------------- */
/* Back end and round-trip through the real reader                           */
/* ------------------------------------------------------------------------- */

/* Reads one 2-bit packed ternary value straight from the format description:
 * 0b00 -> 0, 0b01 -> +1, 0b11 -> -1, two bits per element, low bits first. */
static int decode_ternary(const uint8_t *packed, uint64_t index) {
    uint8_t code = (uint8_t)((packed[index / 4] >> ((index % 4) * 2)) & 0x03u);
    if (code == 0x01) return 1;
    if (code == 0x03) return -1;
    return 0;
}

static void codegen_roundtrip_through_the_reader(void) {
    const char *src =
        "model rt {\n"
        "  layers = 2\n"
        "  hidden = 8\n"
        "  state  = 3\n"
        "  vocab  = 16\n"
        "}\n"
        "for l in 0..2 {\n"
        "  tensor \"layers.${l}.w\" {\n"
        "    dtype = ternary\n"
        "    shape = [10, 8]\n"
        "    scale = 0.125\n"
        "    init  = ternary\n"
        "  }\n"
        "}\n"
        "tensor bias { dtype = fp32 shape = [10] init = ramp }\n";

    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "rt.nano", &p, &d, NULL) == 0, "should compile");
    if (p.n_tensors != 3) {
        CHECK(0, "expected 3 tensors, got %d", p.n_tensors);
        nano_program_free(&p);
        return;
    }
    CHECK(p.tensors[0].data_size == 20, "ternary[10,8]=80 values should be 20 bytes");

    char out[512];
    tpath("rt.nanow", out, sizeof(out));
    nano_diag_t ed;
    nano_diag_init(&ed);
    CHECK(nano_emit_nanow(&p, out, &ed) == 0, "emit should succeed");

    /* Now read it back with phase1's reader, the same code the runtime uses. */
    nanow_model_t model;
    CHECK(nanow_model_open(&model, out) == 0, "the reader should open what we wrote");
    if (!model.is_open) {
        nano_program_free(&p);
        return;
    }

    CHECK(model.header.magic == NANOW_MAGIC, "magic should be set by the writer");
    CHECK(model.header.version == NANOW_VERSION, "version should be set by the writer");
    CHECK(model.header.flags == 0, "a compiled model is not a KV cache, got %#x",
          model.header.flags);
    CHECK(model.header.num_tensors == 3, "the reader should see 3 tensors, got %u",
          model.header.num_tensors);
    CHECK(model.header.num_layers == 2 && model.header.hidden_dim == 8 &&
              model.header.state_dim == 3 && model.header.vocab_size == 16,
          "architecture fields should survive the round trip");
    CHECK(model.header.total_params == 2 * 80 + 10, "total_params should be 170, got %llu",
          (unsigned long long)model.header.total_params);

    const nanow_tensor_desc_t *wd = nanow_model_get_tensor_desc(&model, "layers.1.w");
    CHECK(wd != NULL, "layers.1.w should be found by name");
    if (wd) {
        CHECK(wd->dtype == NANOW_DTYPE_TERNARY, "dtype should be ternary");
        CHECK(wd->packing == NANOW_PACKING_2BIT, "packing should be 2bit");
        CHECK(wd->ndims == 2, "ndims should be 2");
        CHECK(wd->shape[0] == 10 && wd->shape[1] == 8, "shape should be [10, 8]");
        CHECK(wd->num_elements == 80, "num_elements should be 80");
        CHECK(wd->data_size == 20, "data_size should be 20");
        CHECK(wd->scale_factor > 0.1249f && wd->scale_factor < 0.1251f,
              "scale should round-trip as 0.125");

        const uint8_t *bytes =
            (const uint8_t *)nanow_model_get_tensor_data(&model, wd);
        CHECK(bytes != NULL, "the payload pointer should be available");
        if (bytes) {
            int ones = 0, zeros = 0, minus = 0;
            for (uint64_t i = 0; i < wd->num_elements; ++i) {
                int v = decode_ternary(bytes, i);
                if (v == 1) ones++;
                else if (v == 0) zeros++;
                else minus++;
            }
            CHECK(ones + zeros + minus == 80, "every element should decode");
            CHECK(minus > 0 && zeros > 0 && ones > 0,
                  "a generated matrix should contain all three values (got %d/%d/%d)",
                  minus, zeros, ones);
            CHECK(ones + minus < 80, "and should contain some zeros");
        }
    }

    const nanow_tensor_desc_t *bd = nanow_model_get_tensor_desc(&model, "bias");
    CHECK(bd != NULL, "bias should be found");
    if (bd) {
        CHECK(bd->dtype == NANOW_DTYPE_FP32, "bias should be fp32");
        const float *vals = (const float *)nanow_model_get_tensor_data(&model, bd);
        CHECK(vals != NULL, "bias payload should be readable");
        if (vals) {
            CHECK(vals[0] == 0.0f, "ramp should start at 0");
            CHECK(vals[9] > 0.999f && vals[9] < 1.001f, "ramp should end at 1");
        }
    }

    CHECK(nanow_model_get_tensor_desc(&model, "does.not.exist") == NULL,
          "an unknown name should return NULL");

    nanow_model_close(&model);
    nano_program_free(&p);
}

/*
 * A KV cache is a .nanow container that is not a model. The distinction lives
 * in one header bit, so the writer that the runtime actually reads has to carry
 * it; nanow_write_file() is the only writer shared by the compiler, the Python
 * exporter and this test.
 */
static void codegen_kv_cache_flag_round_trips(void) {
    float payload[4] = {0.0f, 1.0f, 2.0f, 3.0f};

    nanow_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.flags        = NANOW_FLAG_KV_CACHE;
    hdr.num_tensors  = 1;
    hdr.num_layers   = 2;
    hdr.total_params = 4;

    nanow_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    snprintf(desc.name, sizeof(desc.name), "k0");
    desc.dtype        = NANOW_DTYPE_FP32;
    desc.packing      = NANOW_PACKING_RAW_INT8;
    desc.ndims        = 1;
    desc.shape[0]     = 4;
    desc.num_elements = 4;
    desc.data_size    = sizeof(payload);

    const void *payloads[1] = {payload};

    char out[512];
    tpath("kv.nanow", out, sizeof(out));
    CHECK(nanow_write_file(out, &hdr, &desc, payloads) == 0,
          "the writer should accept a KV header");

    nanow_model_t model;
    CHECK(nanow_model_open(&model, out) == 0, "the reader should open a KV container");
    if (model.is_open) {
        CHECK(model.header.flags == NANOW_FLAG_KV_CACHE,
              "the KV flag should survive the writer, got %#x", model.header.flags);
        CHECK(model.header.num_layers == 2, "num_layers should survive, got %u",
              model.header.num_layers);
        const nanow_tensor_desc_t *d = nanow_model_get_tensor_desc(&model, "k0");
        CHECK(d != NULL, "k0 should be found by name");
        if (d) {
            const float *v = (const float *)nanow_model_get_tensor_data(&model, d);
            CHECK(v != NULL && v[0] == 0.0f && v[3] == 3.0f,
                  "the payload should be readable verbatim");
        }
        nanow_model_close(&model);
    }

    /* And a model container must not claim to be a cache. */
    char model_out[512];
    tpath("notkv.nanow", model_out, sizeof(model_out));
    hdr.flags = 0;
    CHECK(nanow_write_file(model_out, &hdr, &desc, payloads) == 0, "plain write should work");
    nanow_model_t plain;
    CHECK(nanow_model_open(&plain, model_out) == 0, "plain container should open");
    if (plain.is_open) {
        CHECK(plain.header.flags == 0, "no flag should mean model weights, got %#x",
              plain.header.flags);
        nanow_model_close(&plain);
    }
}

static void codegen_emit_file_size_is_page_aligned(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = fp32 shape = [4] init = ones }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "size.nano", &p, &d, NULL) == 0, "should compile");

    char out[512];
    tpath("size.nanow", out, sizeof(out));
    nano_diag_t ed;
    nano_diag_init(&ed);
    CHECK(nano_emit_nanow(&p, out, &ed) == 0, "emit should succeed");

    /* header (64) + 1 descriptor (120) + 16 payload bytes, each section pushed
     * to the next 4096-byte boundary. */
    FILE *f = fopen(out, "rb");
    CHECK(f != NULL, "the container should exist");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fclose(f);
        CHECK(size == 8192, "expected 8192 bytes, got %ld", size);
    }
    nano_program_free(&p);
}

static void codegen_ir_text_file(void) {
    const char *src =
        "model m { layers = 1 hidden = 4 }\n"
        "tensor w { dtype = ternary shape = [4, 4] scale = 0.1 init = ternary }\n"
        "op ternary_linear \"lin\" { in = 4 out = 4 weight = \"w\" }\n";
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(src, "irtext.nano", &p, &d, NULL) == 0, "should compile");

    char out[512];
    tpath("ir.txt", out, sizeof(out));
    nano_diag_t ed;
    nano_diag_init(&ed);
    CHECK(nano_write_ir_text(&p, out, &ed) == 0, "writing the IR should succeed");

    FILE *f = fopen(out, "rb");
    CHECK(f != NULL, "the IR file should exist");
    if (f) {
        char buf[4096];
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        buf[n] = '\0';
        fclose(f);
        CHECK(strstr(buf, "nano-ir") != NULL, "the dump should name the IR");
        CHECK(strstr(buf, "ternary_linear") != NULL, "the dump should show the op");
    }
    nano_program_free(&p);
}

static void codegen_ir_text_without_ops_fails(void) {
    nano_program_t p;
    nano_diag_t d;
    CHECK(compile_src(kMinimal, "noop.nano", &p, &d, NULL) == 0, "should compile");

    char out[512];
    tpath("noop.txt", out, sizeof(out));
    nano_diag_t ed;
    nano_diag_init(&ed);
    CHECK(nano_write_ir_text(&p, out, &ed) != 0, "there is no graph to write");
    CHECK(diag_has(&ed, "no IR graph"), "should say so plainly");
    nano_program_free(&p);
}

static void compile_file_reports_a_missing_source(void) {
    nano_program_t p;
    nano_diag_t d;
    nano_diag_init(&d);
    memset(&p, 0, sizeof(p));
    char missing[512];
    tpath("does_not_exist.nano", missing, sizeof(missing));
    CHECK(nano_compile_file(missing, NULL, &p, &d) != 0, "should fail");
    CHECK(diag_has(&d, "cannot open"), "should report the missing file");
    nano_program_free(&p);
}

/* ------------------------------------------------------------------------- */

int main(void) {
    char tmpl[] = "/tmp/nanoc_test_XXXXXX";
    if (!mkdtemp(tmpl)) {
        fprintf(stderr, "could not create a temporary directory\n");
        return 2;
    }
    snprintf(g_tmpdir, sizeof(g_tmpdir), "%s", tmpl);

    fputs("nanoc test suite\n", stderr);

    RUN(lexer_basic_kinds);
    RUN(lexer_range_operator);
    RUN(lexer_float_literal);
    RUN(lexer_comments_are_skipped);
    RUN(lexer_string_keeps_interpolation);
    RUN(lexer_string_escapes);
    RUN(lexer_unterminated_string_fails);
    RUN(lexer_unexpected_character_fails);
    RUN(lexer_unterminated_block_comment_fails);

    RUN(parser_model_fields);
    RUN(parser_shape_value);
    RUN(parser_for_body_and_ident_name);
    RUN(parser_for_closes_its_brace);
    RUN(parser_missing_equals_fails);
    RUN(parser_stray_brace_fails);
    RUN(parser_non_integer_loop_range_fails);

    RUN(sema_minimal_ok);
    RUN(sema_payload_size_table);
    RUN(sema_duplicate_tensor_fails);
    RUN(sema_ternary_requires_scale);
    RUN(sema_ternary_scale_must_be_positive);
    RUN(sema_unbound_variable_fails);
    RUN(sema_loop_unrolls_and_interpolates);
    RUN(sema_unknown_field_fails);
    RUN(sema_duplicate_field_fails);
    RUN(sema_missing_model_fails);
    RUN(sema_no_tensors_fails);
    RUN(sema_zero_dimension_fails);
    RUN(sema_long_tensor_name_fails);
    RUN(sema_max_bytes_guard);
    RUN(sema_emit_statement_recorded);

    RUN(sema_ops_lower_into_ir);
    RUN(sema_ssm_dim_must_match_hidden);
    RUN(sema_ternary_linear_weight_checked);
    RUN(sema_ternary_linear_shape_checked);
    RUN(sema_unknown_weight_tensor_fails);
    RUN(sema_unknown_opcode_fails);

    RUN(sema_fp16_is_gated);
    RUN(sema_radix3_is_gated);

    RUN(sema_from_reads_file);
    RUN(sema_from_missing_file_fails);
    RUN(sema_from_wrong_size_fails);
    RUN(sema_source_without_from_fails);
    RUN(sema_seed_requires_ternary_init);
    RUN(sema_ternary_generation_is_deterministic);
    RUN(sema_ternary_seed_follows_the_name);
    RUN(sema_explicit_seed_overrides_the_name);

    RUN(codegen_roundtrip_through_the_reader);
    RUN(codegen_kv_cache_flag_round_trips);
    RUN(codegen_emit_file_size_is_page_aligned);
    RUN(codegen_ir_text_file);
    RUN(codegen_ir_text_without_ops_fails);
    RUN(compile_file_reports_a_missing_source);

    fprintf(stderr, "\n%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
