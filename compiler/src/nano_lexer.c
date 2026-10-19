/*
 * nano_lexer.c — NanoLang tokeniser and shared diagnostics.
 *
 * Part of the NanoLang compiler; see compiler/README.md for the language spec
 * and include/nano_lang.h for the token/AST contracts.
 */

#include "nano_lang.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* Diagnostics                                                               */
/* ------------------------------------------------------------------------- */

void nano_diag_init(nano_diag_t *d) {
    if (!d) return;
    d->count = 0;
    d->overflow = 0;
}

static void diag_push(nano_diag_t *d, int line, int severity, const char *fmt,
                      va_list ap) {
    if (!d) return;
    if (d->count >= NANO_MAX_DIAGS) {
        d->overflow = 1;
        return;
    }
    nano_diag_item_t *it = &d->items[d->count++];
    it->line = line;
    it->severity = severity;
    vsnprintf(it->msg, NANO_DIAG_MSG, fmt, ap);
}

void nano_diag_add(nano_diag_t *d, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    diag_push(d, line, NANO_SEV_ERROR, fmt, ap);
    va_end(ap);
}

void nano_diag_warn(nano_diag_t *d, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    diag_push(d, line, NANO_SEV_WARNING, fmt, ap);
    va_end(ap);
}

int nano_diag_error_count(const nano_diag_t *d) {
    if (!d) return 0;
    int n = 0;
    for (int i = 0; i < d->count; ++i) {
        if (d->items[i].severity == NANO_SEV_ERROR) n++;
    }
    return n;
}

void nano_diag_print(const nano_diag_t *d, const char *path, FILE *out) {
    if (!d || !out) return;
    const char *where = path ? path : "<input>";
    int errors = 0, warnings = 0;
    for (int i = 0; i < d->count; ++i) {
        const nano_diag_item_t *it = &d->items[i];
        int is_error = (it->severity == NANO_SEV_ERROR);
        if (is_error) errors++; else warnings++;
        fprintf(out, "%s:%d: %s: %s\n", where, it->line,
                is_error ? "error" : "warning", it->msg);
    }
    if (d->overflow) {
        fprintf(out, "%s: note: further diagnostics suppressed\n", where);
    }
    if (errors || warnings) {
        fprintf(out, "%s: %d error%s, %d warning%s\n", where, errors,
                errors == 1 ? "" : "s", warnings, warnings == 1 ? "" : "s");
    }
}

/* ------------------------------------------------------------------------- */
/* Token names                                                               */
/* ------------------------------------------------------------------------- */

const char *nano_token_kind_name(nano_token_kind_t k) {
    switch (k) {
        case NANO_TOK_EOF:        return "end of file";
        case NANO_TOK_IDENT:      return "identifier";
        case NANO_TOK_NUMBER:     return "number";
        case NANO_TOK_STRING:     return "string";
        case NANO_TOK_LBRACE:     return "'{'";
        case NANO_TOK_RBRACE:     return "'}'";
        case NANO_TOK_LBRACKET:   return "'['";
        case NANO_TOK_RBRACKET:   return "']'";
        case NANO_TOK_COMMA:      return "','";
        case NANO_TOK_EQUALS:     return "'='";
        case NANO_TOK_DOTDOT:     return "'..'";
    }
    return "unknown";
}

const char *nano_stmt_kind_name(nano_stmt_kind_t k) {
    switch (k) {
        case NANO_ST_MODEL:  return "model";
        case NANO_ST_TENSOR: return "tensor";
        case NANO_ST_OP:     return "op";
        case NANO_ST_FOR:    return "for";
        case NANO_ST_EMIT:   return "emit";
    }
    return "unknown";
}

/* ------------------------------------------------------------------------- */
/* Lexer                                                                     */
/* ------------------------------------------------------------------------- */

#define NANO_TOKEN_CAP 4096
#define NANO_TEXT_CAP  192

static int is_digit(char c) { return c >= '0' && c <= '9'; }

static int is_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

/* Identifiers may contain dots so that `layers.0.in_proj` needs no quotes. */
static int is_ident_char(char c) {
    return is_ident_start(c) || is_digit(c) || c == '.';
}

typedef struct {
    const char   *src;
    size_t        len;
    size_t        pos;
    int           line;
    nano_diag_t  *diag;
    nano_token_t *toks;
    int           n;
    int           cap;
    int           failed;
} lexer_t;

static void push(lexer_t *L, nano_token_kind_t kind, const char *text, int line) {
    if (L->n >= L->cap) {
        nano_diag_add(L->diag, line, "too many tokens (limit %d)", NANO_TOKEN_CAP);
        L->failed = 1;
        return;
    }
    nano_token_t *t = &L->toks[L->n++];
    memset(t, 0, sizeof(*t));
    t->kind = kind;
    t->line = line;
    if (text) {
        size_t n = strlen(text);
        if (n >= sizeof(t->text)) n = sizeof(t->text) - 1;
        memcpy(t->text, text, n);
        t->text[n] = '\0';
    }
}

static void push_text1(lexer_t *L, nano_token_kind_t kind, char c, int line) {
    char buf[2] = { c, '\0' };
    push(L, kind, buf, line);
}

static int scan_number(lexer_t *L) {
    size_t start = L->pos;
    int is_float = 0;

    while (L->pos < L->len && is_digit(L->src[L->pos])) L->pos++;

    /* A '.' is part of a float only when a digit follows; otherwise it is the
     * start of '..', the range operator. */
    if (L->pos + 1 < L->len && L->src[L->pos] == '.' &&
        is_digit(L->src[L->pos + 1])) {
        is_float = 1;
        L->pos++;
        while (L->pos < L->len && is_digit(L->src[L->pos])) L->pos++;
    }

    char buf[NANO_TEXT_CAP];
    size_t n = L->pos - start;
    if (n >= sizeof(buf)) {
        nano_diag_add(L->diag, L->line, "number literal too long");
        L->failed = 1;
        return -1;
    }
    memcpy(buf, L->src + start, n);
    buf[n] = '\0';

    push(L, NANO_TOK_NUMBER, buf, L->line);
    nano_token_t *t = &L->toks[L->n - 1];
    t->number = strtod(buf, NULL);
    t->is_float = is_float;
    return 0;
}

static int scan_ident(lexer_t *L) {
    size_t start = L->pos;
    while (L->pos < L->len && is_ident_char(L->src[L->pos])) L->pos++;

    size_t n = L->pos - start;
    if (n >= NANO_TEXT_CAP) {
        nano_diag_add(L->diag, L->line, "identifier too long (limit %d chars)",
                      NANO_TEXT_CAP - 1);
        L->failed = 1;
        return -1;
    }
    char buf[NANO_TEXT_CAP];
    memcpy(buf, L->src + start, n);
    buf[n] = '\0';
    push(L, NANO_TOK_IDENT, buf, L->line);
    return 0;
}

/* Strings keep ${...} verbatim; nano_sema.c expands it once loop variables are
 * known. \" \\ \n \t \$ are decoded here, any other backslash is literal. */
static int scan_string(lexer_t *L) {
    int start_line = L->line;
    L->pos++;                       /* opening quote */

    char buf[NANO_TEXT_CAP];
    size_t n = 0;

    while (1) {
        if (L->pos >= L->len) {
            nano_diag_add(L->diag, start_line, "unterminated string literal");
            L->failed = 1;
            return -1;
        }
        char c = L->src[L->pos];

        if (c == '\n') {
            nano_diag_add(L->diag, start_line,
                          "unterminated string literal (newline before closing quote)");
            L->failed = 1;
            return -1;
        }
        if (c == '"') { L->pos++; break; }

        if (c == '\\' && L->pos + 1 < L->len) {
            char e = L->src[L->pos + 1];
            char decoded = -1;
            switch (e) {
                case '"':  decoded = '"';  break;
                case '\\': decoded = '\\'; break;
                case 'n':  decoded = '\n'; break;
                case 't':  decoded = '\t'; break;
                case '$':  decoded = '$';  break;
                default:   break;
            }
            if (decoded != -1) {
                if (n + 1 >= sizeof(buf)) goto too_long;
                buf[n++] = decoded;
                L->pos += 2;
                continue;
            }
            /* Unknown escape: keep the backslash as written. */
        }

        if (n + 1 >= sizeof(buf)) goto too_long;
        buf[n++] = c;
        L->pos++;
    }

    buf[n] = '\0';
    push(L, NANO_TOK_STRING, buf, start_line);
    return 0;

too_long:
    nano_diag_add(L->diag, start_line, "string literal too long (limit %d chars)",
                  NANO_TEXT_CAP - 1);
    L->failed = 1;
    return -1;
}

static int skip_trivia(lexer_t *L) {
    while (L->pos < L->len) {
        char c = L->src[L->pos];

        if (c == '\n') { L->line++; L->pos++; continue; }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f') {
            L->pos++;
            continue;
        }
        if (c == '/' && L->pos + 1 < L->len && L->src[L->pos + 1] == '/') {
            while (L->pos < L->len && L->src[L->pos] != '\n') L->pos++;
            continue;
        }
        if (c == '/' && L->pos + 1 < L->len && L->src[L->pos + 1] == '*') {
            int start_line = L->line;
            L->pos += 2;
            int closed = 0;
            while (L->pos < L->len) {
                if (L->src[L->pos] == '\n') L->line++;
                if (L->src[L->pos] == '*' && L->pos + 1 < L->len &&
                    L->src[L->pos + 1] == '/') {
                    L->pos += 2;
                    closed = 1;
                    break;
                }
                L->pos++;
            }
            if (!closed) {
                nano_diag_add(L->diag, start_line, "unterminated block comment");
                L->failed = 1;
                return -1;
            }
            continue;
        }
        break;
    }
    return 0;
}

int nano_lex(const char *src, size_t len, nano_token_t **out, int *count,
             nano_diag_t *diag) {
    if (!src || !out || !count) return -1;
    *out = NULL;
    *count = 0;

    lexer_t L;
    memset(&L, 0, sizeof(L));
    L.src = src;
    L.len = len;
    L.pos = 0;
    L.line = 1;
    L.diag = diag;
    L.cap = NANO_TOKEN_CAP;
    L.toks = (nano_token_t *)malloc(sizeof(nano_token_t) * (size_t)L.cap);
    if (!L.toks) {
        nano_diag_add(diag, 0, "out of memory while lexing");
        return -1;
    }

    while (L.pos < L.len) {
        if (skip_trivia(&L) != 0) break;

        if (L.pos >= L.len) break;
        char c = L.src[L.pos];

        if (c == '{')      { push_text1(&L, NANO_TOK_LBRACE, c, L.line); L.pos++; continue; }
        if (c == '}')      { push_text1(&L, NANO_TOK_RBRACE, c, L.line); L.pos++; continue; }
        if (c == '[')      { push_text1(&L, NANO_TOK_LBRACKET, c, L.line); L.pos++; continue; }
        if (c == ']')      { push_text1(&L, NANO_TOK_RBRACKET, c, L.line); L.pos++; continue; }
        if (c == ',')      { push_text1(&L, NANO_TOK_COMMA, c, L.line); L.pos++; continue; }
        if (c == '=')      { push_text1(&L, NANO_TOK_EQUALS, c, L.line); L.pos++; continue; }
        if (c == '.') {
            if (L.pos + 1 < L.len && L.src[L.pos + 1] == '.') {
                push_text1(&L, NANO_TOK_DOTDOT, '.', L.line);
                L.pos += 2;
                continue;
            }
            nano_diag_add(diag, L.line,
                          "unexpected '.' (did you mean the range operator '..'?)");
            L.failed = 1;
            L.pos++;
            continue;
        }
        if (is_digit(c))    { if (scan_number(&L) != 0) break; continue; }
        if (is_ident_start(c)) { if (scan_ident(&L) != 0) break; continue; }
        if (c == '"')       { if (scan_string(&L) != 0) break; continue; }

        nano_diag_add(diag, L.line, "unexpected character '%c' (0x%02x)",
                      (c >= 32 && c < 127) ? c : '?', (unsigned char)c);
        L.failed = 1;
        L.pos++;
    }

    if (L.failed) {
        free(L.toks);
        return -1;
    }

    push(&L, NANO_TOK_EOF, NULL, L.line);
    *out = L.toks;
    *count = L.n;
    return 0;
}
