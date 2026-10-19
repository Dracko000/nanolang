/*
 * nano_parser.c — recursive-descent parser for NanoLang.
 *
 * Grammar (see compiler/README.md for the annotated version):
 *
 *   program := stmt*
 *   stmt    := model_stmt | tensor_stmt | op_stmt | for_stmt | emit_stmt
 *   model   := "model"  name "{" field* "}"
 *   tensor  := "tensor" name "{" field* "}"
 *   op      := "op" ident name "{" field* "}"
 *   for     := "for" ident "in" NUMBER ".." NUMBER "{" stmt* "}"
 *   emit    := "emit" STRING
 *   field   := IDENT "=" value
 *   value   := NUMBER | STRING | IDENT | "[" NUMBER ("," NUMBER)* "]"
 *
 * The parser only builds syntax. Name interpolation, field validity, tensor
 * consistency and graph construction all happen in nano_sema.c.
 */

#include "nano_lang.h"

#include <stdlib.h>
#include <string.h>

#define NANO_MAX_DIMS 4

typedef struct {
    const nano_token_t *toks;
    int                 n;
    int                 pos;
    nano_diag_t        *diag;
    int                 failed;
} parser_t;

/* ------------------------------------------------------------------------- */
/* Token helpers                                                             */
/* ------------------------------------------------------------------------- */

static const nano_token_t *peek(const parser_t *P) { return &P->toks[P->pos]; }

static const nano_token_t *advance(parser_t *P) {
    const nano_token_t *t = &P->toks[P->pos];
    if (P->pos < P->n - 1) P->pos++;
    return t;
}

static int at_kind(const parser_t *P, nano_token_kind_t k) {
    return peek(P)->kind == k;
}

static int at_keyword(const parser_t *P, const char *kw) {
    return peek(P)->kind == NANO_TOK_IDENT && strcmp(peek(P)->text, kw) == 0;
}

/* Reports and consumes the unexpected token, then keeps going so that several
 * syntax errors surface in one run rather than one per invocation. */
static void unexpected(parser_t *P, const char *what) {
    const nano_token_t *t = peek(P);
    char detail[256];
    if (t->kind == NANO_TOK_EOF) {
        snprintf(detail, sizeof(detail), "unexpected end of file, expected %s", what);
    } else if (t->kind == NANO_TOK_IDENT || t->kind == NANO_TOK_NUMBER ||
               t->kind == NANO_TOK_STRING) {
        snprintf(detail, sizeof(detail), "unexpected %s '%.96s', expected %s",
                 nano_token_kind_name(t->kind), t->text, what);
    } else {
        snprintf(detail, sizeof(detail), "unexpected %s, expected %s",
                 nano_token_kind_name(t->kind), what);
    }
    nano_diag_add(P->diag, t->line, "%s", detail);
    P->failed = 1;
    if (t->kind != NANO_TOK_EOF) advance(P);
}

static int expect(parser_t *P, nano_token_kind_t k, const char *what) {
    if (!at_kind(P, k)) {
        unexpected(P, what);
        return -1;
    }
    advance(P);
    return 0;
}

static int expect_keyword(parser_t *P, const char *kw) {
    if (!at_keyword(P, kw)) {
        unexpected(P, kw);
        return -1;
    }
    advance(P);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Growable statement list                                                   */
/* ------------------------------------------------------------------------- */

typedef struct {
    nano_stmt_t *items;
    int          n;
    int          cap;
} stmt_list_t;

static int stmt_list_push(stmt_list_t *L, const nano_stmt_t *v) {
    if (L->n >= L->cap) {
        int cap = L->cap ? L->cap * 2 : 8;
        nano_stmt_t *grown =
            (nano_stmt_t *)realloc(L->items, sizeof(nano_stmt_t) * (size_t)cap);
        if (!grown) return -1;
        L->items = grown;
        L->cap = cap;
    }
    L->items[L->n++] = *v;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Names and values                                                          */
/* ------------------------------------------------------------------------- */

/* A name is an identifier or a quoted string, so that `"layers.${layer}.w"`
 * and `embed_tokens` are both accepted where a name is expected. */
static int parse_name(parser_t *P, char *out, size_t out_cap) {
    if (at_kind(P, NANO_TOK_IDENT) || at_kind(P, NANO_TOK_STRING)) {
        const nano_token_t *t = advance(P);
        snprintf(out, out_cap, "%s", t->text);
        return 0;
    }
    unexpected(P, "a name (identifier or quoted string)");
    return -1;
}

static int parse_value(parser_t *P, nano_field_t *f) {
    if (at_kind(P, NANO_TOK_NUMBER)) {
        const nano_token_t *t = advance(P);
        f->num = t->number;
        f->kind = t->is_float ? NANO_VAL_FLOAT : NANO_VAL_INT;
        return 0;
    }
    if (at_kind(P, NANO_TOK_STRING)) {
        const nano_token_t *t = advance(P);
        snprintf(f->str, sizeof(f->str), "%s", t->text);
        f->kind = NANO_VAL_STRING;
        return 0;
    }
    if (at_kind(P, NANO_TOK_IDENT)) {
        const nano_token_t *t = advance(P);
        snprintf(f->str, sizeof(f->str), "%s", t->text);
        f->kind = NANO_VAL_IDENT;
        return 0;
    }
    if (at_kind(P, NANO_TOK_LBRACKET)) {
        const nano_token_t *open = advance(P);
        f->kind = NANO_VAL_SHAPE;
        f->n_shape = 0;
        if (at_kind(P, NANO_TOK_RBRACKET)) {
            nano_diag_add(P->diag, open->line, "empty shape list; expected at least one dimension");
            P->failed = 1;
            advance(P);
            return -1;
        }
        while (1) {
            if (!at_kind(P, NANO_TOK_NUMBER)) {
                unexpected(P, "a dimension (positive integer)");
                return -1;
            }
            const nano_token_t *t = advance(P);
            if (t->is_float || t->number < 0 || t->number > 2147483647.0) {
                nano_diag_add(P->diag, t->line,
                              "dimension '%s' must be a positive integer", t->text);
                P->failed = 1;
            } else if (f->n_shape >= NANO_MAX_DIMS) {
                nano_diag_add(P->diag, t->line,
                              "too many dimensions (limit %d)", NANO_MAX_DIMS);
                P->failed = 1;
            } else {
                f->shape[f->n_shape++] = (int)t->number;
            }
            if (at_kind(P, NANO_TOK_COMMA)) { advance(P); continue; }
            break;
        }
        if (expect(P, NANO_TOK_RBRACKET, "']'") != 0) return -1;
        return 0;
    }
    unexpected(P, "a value (number, string, identifier or [shape])");
    return -1;
}

/* ------------------------------------------------------------------------- */
/* Statements                                                                */
/* ------------------------------------------------------------------------- */

static int parse_stmt_list(parser_t *P, stmt_list_t *out, int inside_block);

static int parse_field_block(parser_t *P, nano_stmt_t *st) {
    /* Caller consumed the opening '{'. */
    int cap = 0;
    while (!at_kind(P, NANO_TOK_RBRACE) && !at_kind(P, NANO_TOK_EOF)) {
        if (!at_kind(P, NANO_TOK_IDENT)) {
            unexpected(P, "a field name");
            if (at_kind(P, NANO_TOK_EOF)) break;
            continue;
        }
        nano_field_t f;
        memset(&f, 0, sizeof(f));

        const nano_token_t *key = advance(P);
        if (strlen(key->text) >= sizeof(f.key)) {
            nano_diag_add(P->diag, key->line,
                          "field name is too long (limit %d characters)",
                          (int)sizeof(f.key) - 1);
            P->failed = 1;
        }
        snprintf(f.key, sizeof(f.key), "%.*s", (int)sizeof(f.key) - 1, key->text);
        f.line = key->line;

        if (expect(P, NANO_TOK_EQUALS, "'=' after the field name") != 0) {
            /* Resynchronise: skip whatever this field was. */
            while (!at_kind(P, NANO_TOK_RBRACE) && !at_kind(P, NANO_TOK_EOF) &&
                   !at_kind(P, NANO_TOK_IDENT)) {
                advance(P);
            }
            continue;
        }
        if (parse_value(P, &f) != 0) {
            while (!at_kind(P, NANO_TOK_RBRACE) && !at_kind(P, NANO_TOK_EOF) &&
                   !at_kind(P, NANO_TOK_IDENT)) {
                advance(P);
            }
            continue;
        }

        if (st->n_fields >= cap) {
            int new_cap = cap ? cap * 2 : 8;
            nano_field_t *grown = (nano_field_t *)realloc(
                st->fields, sizeof(nano_field_t) * (size_t)new_cap);
            if (!grown) {
                nano_diag_add(P->diag, f.line, "out of memory while parsing fields");
                P->failed = 1;
                return -1;
            }
            st->fields = grown;
            cap = new_cap;
        }
        st->fields[st->n_fields++] = f;
    }
    if (expect(P, NANO_TOK_RBRACE, "'}' to close the block") != 0) return -1;
    return 0;
}

static int parse_one(parser_t *P, nano_stmt_t *st) {
    memset(st, 0, sizeof(*st));

    const nano_token_t *kw = peek(P);
    st->line = kw->line;

    if (at_keyword(P, "model")) {
        advance(P);
        st->kind = NANO_ST_MODEL;
        if (parse_name(P, st->name, sizeof(st->name)) != 0) return -1;
        if (expect(P, NANO_TOK_LBRACE, "'{' after the model name") != 0) return -1;
        return parse_field_block(P, st);
    }
    if (at_keyword(P, "tensor")) {
        advance(P);
        st->kind = NANO_ST_TENSOR;
        if (parse_name(P, st->name, sizeof(st->name)) != 0) return -1;
        if (expect(P, NANO_TOK_LBRACE, "'{' after the tensor name") != 0) return -1;
        return parse_field_block(P, st);
    }
    if (at_keyword(P, "op")) {
        advance(P);
        st->kind = NANO_ST_OP;
        if (!at_kind(P, NANO_TOK_IDENT)) {
            unexpected(P, "an opcode name");
            return -1;
        }
        const nano_token_t *opc = advance(P);
        if (strlen(opc->text) >= sizeof(st->opcode)) {
            nano_diag_add(P->diag, opc->line,
                          "opcode name is too long (limit %d characters)",
                          (int)sizeof(st->opcode) - 1);
            P->failed = 1;
        }
        snprintf(st->opcode, sizeof(st->opcode), "%.*s",
                 (int)sizeof(st->opcode) - 1, opc->text);
        if (parse_name(P, st->name, sizeof(st->name)) != 0) return -1;
        if (expect(P, NANO_TOK_LBRACE, "'{' after the op name") != 0) return -1;
        return parse_field_block(P, st);
    }
    if (at_keyword(P, "for")) {
        advance(P);
        st->kind = NANO_ST_FOR;
        if (!at_kind(P, NANO_TOK_IDENT)) {
            unexpected(P, "a loop variable name");
            return -1;
        }
        const nano_token_t *var = advance(P);
        if (strlen(var->text) >= sizeof(st->var)) {
            nano_diag_add(P->diag, var->line,
                          "loop variable name is too long (limit %d characters)",
                          (int)sizeof(st->var) - 1);
            P->failed = 1;
        }
        snprintf(st->var, sizeof(st->var), "%.*s", (int)sizeof(st->var) - 1,
                 var->text);
        if (expect_keyword(P, "in") != 0) return -1;

        if (!at_kind(P, NANO_TOK_NUMBER)) {
            unexpected(P, "a start integer");
            return -1;
        }
        const nano_token_t *lo = advance(P);
        if (lo->is_float) {
            nano_diag_add(P->diag, lo->line, "loop start '%s' must be an integer", lo->text);
            P->failed = 1;
        }
        if (expect(P, NANO_TOK_DOTDOT, "'..' in the loop range") != 0) return -1;

        if (!at_kind(P, NANO_TOK_NUMBER)) {
            unexpected(P, "an end integer");
            return -1;
        }
        const nano_token_t *hi = advance(P);
        if (hi->is_float) {
            nano_diag_add(P->diag, hi->line, "loop end '%s' must be an integer", hi->text);
            P->failed = 1;
        }
        st->lo = (int)lo->number;
        st->hi = (int)hi->number;

        if (expect(P, NANO_TOK_LBRACE, "'{' after the loop range") != 0) return -1;
        stmt_list_t body;
        memset(&body, 0, sizeof(body));
        if (parse_stmt_list(P, &body, 1) != 0) {
            free(body.items);
            return -1;
        }
        if (expect(P, NANO_TOK_RBRACE, "'}' to close the loop body") != 0) {
            free(body.items);
            return -1;
        }
        st->body = body.items;
        st->n_body = body.n;
        return 0;
    }
    if (at_keyword(P, "emit")) {
        advance(P);
        st->kind = NANO_ST_EMIT;
        if (!at_kind(P, NANO_TOK_STRING)) {
            unexpected(P, "a quoted output path");
            return -1;
        }
        const nano_token_t *t = advance(P);
        snprintf(st->name, sizeof(st->name), "%s", t->text);
        return 0;
    }

    unexpected(P, "a statement (model, tensor, op, for or emit)");
    return -1;
}

static int parse_stmt_list(parser_t *P, stmt_list_t *out, int inside_block) {
    while (!at_kind(P, NANO_TOK_EOF)) {
        if (inside_block && at_kind(P, NANO_TOK_RBRACE)) break;
        if (!inside_block && at_kind(P, NANO_TOK_RBRACE)) {
            unexpected(P, "a statement");
            continue;
        }
        nano_stmt_t st;
        if (parse_one(P, &st) != 0) {
            /* parse_one may have produced a partial statement; drop it. */
            free(st.fields);
            free(st.body);
            if (at_kind(P, NANO_TOK_EOF)) break;
        } else {
            if (stmt_list_push(out, &st) != 0) {
                nano_diag_add(P->diag, st.line, "out of memory while parsing");
                P->failed = 1;
                free(st.fields);
                free(st.body);
                return -1;
            }
        }
    }
    return 0;
}

static void stmt_free(nano_stmt_t *st) {
    free(st->fields);
    for (int i = 0; i < st->n_body; ++i) stmt_free(&st->body[i]);
    free(st->body);
    st->fields = NULL;
    st->body = NULL;
    st->n_fields = 0;
    st->n_body = 0;
}

void nano_ast_free(nano_ast_t *ast) {
    if (!ast) return;
    for (int i = 0; i < ast->n_stmts; ++i) stmt_free(&ast->stmts[i]);
    free(ast->stmts);
    free(ast);
}

int nano_parse(const nano_token_t *tokens, int count, nano_ast_t **out,
               nano_diag_t *diag) {
    if (!tokens || !out) return -1;
    *out = NULL;

    parser_t P;
    memset(&P, 0, sizeof(P));
    P.toks = tokens;
    P.n = count;
    P.pos = 0;
    P.diag = diag;

    stmt_list_t list;
    memset(&list, 0, sizeof(list));
    if (parse_stmt_list(&P, &list, 0) != 0) {
        for (int i = 0; i < list.n; ++i) stmt_free(&list.items[i]);
        free(list.items);
        return -1;
    }
    if (P.failed) {
        for (int i = 0; i < list.n; ++i) stmt_free(&list.items[i]);
        free(list.items);
        return -1;
    }

    nano_ast_t *ast = (nano_ast_t *)calloc(1, sizeof(nano_ast_t));
    if (!ast) {
        for (int i = 0; i < list.n; ++i) stmt_free(&list.items[i]);
        free(list.items);
        nano_diag_add(diag, 0, "out of memory while building the AST");
        return -1;
    }
    ast->stmts = list.items;
    ast->n_stmts = list.n;
    *out = ast;
    return 0;
}
