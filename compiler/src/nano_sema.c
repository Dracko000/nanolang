/*
 * nano_sema.c — NanoLang semantic analysis and lowering.
 *
 * Responsibilities, in order:
 *   1. unroll `for` blocks, binding the loop variable
 *   2. expand ${var} inside names, rejecting unbound references
 *   3. validate field names, duplicates and value types
 *   4. build one payload buffer per `tensor`, checking sizes and total cost
 *   5. lower `op` statements into the phase1 nano-ir graph and validate it
 *
 * Design note: the language deliberately cannot emit a container the runtime
 * cannot read. The runtime (run_smollm2.py, phase2/src/nanolang_engine.c) only
 * implements 2-bit ternary packing, so requests for radix3 or raw ternary are
 * rejected here rather than silently producing an unreadable file.
 */

#include "nano_lang.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* phase1/src/nanow_writer.c */
extern size_t pack_ternary_2bit(uint8_t *out_bytes, const int8_t *in_ternary,
                                size_t count);
extern size_t pack_ternary_radix3(uint8_t *out_bytes, const int8_t *in_ternary,
                                  size_t count);

#define NANO_MAX_ENV   8
#define NANO_PATH_CAP  512
#define NANO_MAX_DIM_VAL 1048576u   /* guard against typo-sized shapes */

/* ------------------------------------------------------------------------- */
/* Payload sizing                                                            */
/* ------------------------------------------------------------------------- */

size_t nano_payload_bytes(uint32_t dtype, uint32_t packing, uint64_t num_elements) {
    switch (dtype) {
        case NANOW_DTYPE_TERNARY:
            if (packing == NANOW_PACKING_2BIT)
                return (size_t)((num_elements + 3) / 4);
            if (packing == NANOW_PACKING_RAW_INT8)
                return (size_t)num_elements;
            if (packing == NANOW_PACKING_RADIX3)
                return (size_t)((num_elements + 4) / 5);
            return 0;
        case NANOW_DTYPE_INT8:
            return (size_t)num_elements;
        case NANOW_DTYPE_FP16:
            return (size_t)(num_elements * 2);
        case NANOW_DTYPE_FP32:
            return (size_t)(num_elements * 4);
        default:
            return 0;
    }
}

/* ------------------------------------------------------------------------- */
/* Half-precision conversion (needed for fp16 `ones` / `ramp` initialisers)  */
/* ------------------------------------------------------------------------- */

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t biased = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (biased == 0xFFu) {                       /* inf or NaN */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }

    int exp = (int)biased - 127 + 15;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);   /* overflow -> inf */

    if (exp <= 0) {                              /* subnormal or zero */
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t m = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) m++;
        return (uint16_t)(sign | m);
    }

    uint32_t m = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
        m++;
        if (m == 0x400u) {
            m = 0;
            if (++exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
}

/* ------------------------------------------------------------------------- */
/* Loop-variable environment                                                 */
/* ------------------------------------------------------------------------- */

typedef struct {
    const char *name;
    int         value;
} nano_env_slot_t;

typedef struct {
    nano_env_slot_t slots[NANO_MAX_ENV];
    int             n;
} nano_env_t;

static int env_lookup(const nano_env_t *e, const char *name, int *out) {
    for (int i = e->n - 1; i >= 0; --i) {
        if (strcmp(e->slots[i].name, name) == 0) {
            *out = e->slots[i].value;
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Sema state                                                                */
/* ------------------------------------------------------------------------- */

typedef struct {
    nano_diag_t         *diag;
    const nano_options_t *opt;
    nano_program_t      *prog;

    int failed;
    int has_model;
    int n_emit;
    int n_pending_ops;

    uint64_t payload_bytes;
    int      tensor_cap;

    /* `op` nodes are buffered until the model metadata is known, because the
     * nano-ir graph needs hidden_dim/state_dim/num_layers at init time. */
    nano_ir_node_t *pending_ops;
    int             pending_cap;

    char base_dir[NANO_PATH_CAP];
} sema_t;

static void fail(sema_t *S, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char msg[NANO_DIAG_MSG];
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    nano_diag_add(S->diag, line, "%s", msg);
    S->failed = 1;
}

/* Flags something the container format allows but no reader in this repository
 * can decode. By default that is an error, so the compiler cannot silently
 * produce a container that only partly loads; --allow-unreadable downgrades it
 * to a warning for authors who have a consumer we do not know about.
 * Returns 0 to continue, -1 when the caller should skip the declaration. */
static int unreadable(sema_t *S, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char msg[NANO_DIAG_MSG];
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    if (S->opt->allow_unreadable) {
        nano_diag_warn(S->diag, line, "%s", msg);
        return 0;
    }
    nano_diag_add(S->diag, line, "%s (or pass --allow-unreadable to emit it anyway)", msg);
    S->failed = 1;
    return -1;
}

/* ------------------------------------------------------------------------- */
/* Fields                                                                    */
/* ------------------------------------------------------------------------- */

static const nano_field_t *field_find(const nano_stmt_t *st, const char *key) {
    for (int i = 0; i < st->n_fields; ++i) {
        if (strcmp(st->fields[i].key, key) == 0) return &st->fields[i];
    }
    return NULL;
}

static int key_in_list(const char *key, const char *const *keys, int n) {
    for (int i = 0; i < n; ++i) {
        if (strcmp(key, keys[i]) == 0) return 1;
    }
    return 0;
}

static void check_fields(sema_t *S, const nano_stmt_t *st,
                         const char *const *keys, int n_keys) {
    for (int i = 0; i < st->n_fields; ++i) {
        const nano_field_t *f = &st->fields[i];
        if (!key_in_list(f->key, keys, n_keys)) {
            fail(S, f->line, "unknown field '%s' in %s block", f->key,
                 nano_stmt_kind_name(st->kind));
            continue;
        }
        for (int j = i + 1; j < st->n_fields; ++j) {
            if (strcmp(f->key, st->fields[j].key) == 0) {
                fail(S, st->fields[j].line, "duplicate field '%s'", f->key);
            }
        }
    }
}

static int field_as_int(sema_t *S, const nano_field_t *f, const char *what,
                        int *out) {
    if (f->kind == NANO_VAL_INT) {
        *out = (int)f->num;
        return 0;
    }
    if (f->kind == NANO_VAL_FLOAT) {
        fail(S, f->line, "%s must be an integer, got a fractional value", what);
        return -1;
    }
    fail(S, f->line, "%s must be an integer", what);
    return -1;
}

static int get_int_field(sema_t *S, const nano_stmt_t *st, const char *key,
                         int required, int def, int *out) {
    const nano_field_t *f = field_find(st, key);
    if (!f) {
        if (required) {
            fail(S, st->line, "%s block is missing required field '%s'",
                 nano_stmt_kind_name(st->kind), key);
            return -1;
        }
        *out = def;
        return 0;
    }
    return field_as_int(S, f, key, out);
}

static int get_float_field(sema_t *S, const nano_stmt_t *st, const char *key,
                           int required, float def, float *out, int *present) {
    const nano_field_t *f = field_find(st, key);
    if (present) *present = f != NULL;
    if (!f) {
        if (required) {
            fail(S, st->line, "%s block is missing required field '%s'",
                 nano_stmt_kind_name(st->kind), key);
            return -1;
        }
        *out = def;
        return 0;
    }
    if (f->kind == NANO_VAL_INT || f->kind == NANO_VAL_FLOAT) {
        *out = (float)f->num;
        return 0;
    }
    fail(S, f->line, "field '%s' must be a number", key);
    return -1;
}

/* ------------------------------------------------------------------------- */
/* Name interpolation                                                        */
/* ------------------------------------------------------------------------- */

static int expand_name(sema_t *S, const char *raw, const nano_env_t *env,
                       char *out, size_t cap, int line) {
    size_t o = 0;

    for (size_t i = 0; raw[i];) {
        if (raw[i] == '$' && raw[i + 1] == '{') {
            size_t j = i + 2;
            char var[64];
            size_t vn = 0;
            while (raw[j] && raw[j] != '}') {
                if (vn + 1 < sizeof(var)) var[vn++] = raw[j];
                j++;
            }
            var[vn] = '\0';

            if (raw[j] != '}') {
                fail(S, line, "unterminated '${' in name \"%s\"", raw);
                return -1;
            }
            if (vn == 0) {
                fail(S, line, "empty '${}' in name \"%s\"", raw);
                return -1;
            }

            int val;
            if (!env_lookup(env, var, &val)) {
                fail(S, line,
                     "unbound loop variable '${%s}' in name \"%s\"", var, raw);
                return -1;
            }

            char num[24];
            snprintf(num, sizeof(num), "%d", val);
            for (size_t k = 0; num[k]; ++k) {
                if (o + 1 >= cap) goto too_long;
                out[o++] = num[k];
            }
            i = j + 1;
            continue;
        }

        if (o + 1 >= cap) goto too_long;
        out[o++] = raw[i++];
    }

    out[o] = '\0';
    return 0;

too_long:
    fail(S, line, "expanded name is too long (limit %zu bytes): \"%s\"",
         cap - 1, raw);
    return -1;
}

/* ------------------------------------------------------------------------- */
/* Model block                                                               */
/* ------------------------------------------------------------------------- */

static const char *const MODEL_KEYS[] = { "layers", "hidden", "state", "vocab" };

static void handle_model(sema_t *S, const nano_stmt_t *st) {
    if (S->has_model) {
        fail(S, st->line, "duplicate model block; only one is allowed");
        return;
    }
    S->has_model = 1;
    check_fields(S, st, MODEL_KEYS, (int)(sizeof(MODEL_KEYS) / sizeof(*MODEL_KEYS)));

    int layers = 0, hidden = 0, state = 0, vocab = 0;
    int ok = 1;
    ok &= get_int_field(S, st, "layers", 1, 0, &layers) == 0;
    ok &= get_int_field(S, st, "hidden", 1, 0, &hidden) == 0;
    ok &= get_int_field(S, st, "state",  0, 0, &state)  == 0;
    ok &= get_int_field(S, st, "vocab",  0, 0, &vocab)  == 0;
    if (!ok) return;

    if (layers <= 0) { fail(S, st->line, "layers must be greater than zero (got %d)", layers); ok = 0; }
    if (hidden <= 0) { fail(S, st->line, "hidden must be greater than zero (got %d)", hidden); ok = 0; }
    if (hidden > (int)NANO_MAX_DIM_VAL) { fail(S, st->line, "hidden %d exceeds the supported maximum %u", hidden, NANO_MAX_DIM_VAL); ok = 0; }
    if (state < 0)   { fail(S, st->line, "state must not be negative (got %d)", state); ok = 0; }
    if (vocab < 0)   { fail(S, st->line, "vocab must not be negative (got %d)", vocab); ok = 0; }
    if (!ok) return;

    S->prog->num_layers = (uint32_t)layers;
    S->prog->hidden_dim = (uint32_t)hidden;
    S->prog->state_dim  = (uint32_t)state;
    S->prog->vocab_size = (uint32_t)vocab;
}

/* ------------------------------------------------------------------------- */
/* Tensor block                                                              */
/* ------------------------------------------------------------------------- */

static const char *const TENSOR_KEYS[] = {
    "dtype", "shape", "init", "packing", "scale", "seed", "source"
};

typedef enum {
    INIT_NONE = -1,
    INIT_ZEROS = 0,
    INIT_ONES,
    INIT_RAMP,
    INIT_TERNARY,
    INIT_FROM
} init_kind_t;

static int dtype_from_name(const char *s, uint32_t *dtype) {
    if (strcmp(s, "ternary") == 0) { *dtype = NANOW_DTYPE_TERNARY; return 0; }
    if (strcmp(s, "int8") == 0)    { *dtype = NANOW_DTYPE_INT8;    return 0; }
    if (strcmp(s, "fp16") == 0)    { *dtype = NANOW_DTYPE_FP16;    return 0; }
    if (strcmp(s, "fp32") == 0)    { *dtype = NANOW_DTYPE_FP32;    return 0; }
    return -1;
}

static int init_from_name(const char *s, init_kind_t *init) {
    if (strcmp(s, "zeros") == 0)   { *init = INIT_ZEROS;   return 0; }
    if (strcmp(s, "ones") == 0)    { *init = INIT_ONES;    return 0; }
    if (strcmp(s, "ramp") == 0)    { *init = INIT_RAMP;    return 0; }
    if (strcmp(s, "ternary") == 0) { *init = INIT_TERNARY; return 0; }
    if (strcmp(s, "from") == 0)    { *init = INIT_FROM;    return 0; }
    return -1;
}

static nano_tensor_t *find_tensor(nano_program_t *p, const char *name) {
    for (int i = 0; i < p->n_tensors; ++i) {
        if (strcmp(p->tensors[i].name, name) == 0) return &p->tensors[i];
    }
    return NULL;
}

/* FNV-1a. Used to seed `init = ternary` from the tensor name, so a loop that
 * declares the same template for layers 0..3 still produces four different
 * weight matrices, and recompiling the same source reproduces them exactly. */
static uint32_t name_hash(const char *s) {
    uint32_t h = 2166136261u;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        h ^= (uint32_t)*p;
        h *= 16777619u;
    }
    return h;
}

static int push_tensor(sema_t *S, const nano_tensor_t *t) {
    nano_program_t *p = S->prog;
    if (p->n_tensors >= S->tensor_cap) {
        int cap = S->tensor_cap ? S->tensor_cap * 2 : 16;
        nano_tensor_t *grown = (nano_tensor_t *)realloc(
            p->tensors, sizeof(nano_tensor_t) * (size_t)cap);
        if (!grown) {
            fail(S, t->line, "out of memory while adding tensor '%s'", t->name);
            return -1;
        }
        p->tensors = grown;
        S->tensor_cap = cap;
    }
    p->tensors[p->n_tensors++] = *t;
    return 0;
}

static int resolve_path(sema_t *S, const char *rel, char *out, size_t cap,
                        int line) {
    const char *prefix = NULL;
    if (rel[0] != '/' && S->base_dir[0] != '\0') prefix = S->base_dir;

    /* memcpy rather than snprintf so the size check and the copy cannot
     * disagree. */
    size_t rel_len = strlen(rel);
    size_t pre_len = prefix ? strlen(prefix) : 0;
    if (rel_len + (pre_len ? pre_len + 1 : 0) >= cap) {
        fail(S, line, "path is too long for the %zu-byte buffer: '%s%s%s'", cap,
             prefix ? prefix : "", prefix ? "/" : "", rel);
        return -1;
    }

    size_t o = 0;
    if (prefix) {
        memcpy(out, prefix, pre_len);
        o = pre_len;
        out[o++] = '/';
    }
    memcpy(out + o, rel, rel_len + 1);
    return 0;
}

/* Reads exactly `want` bytes; a short or over-long file is an error because a
 * silently truncated weight tensor is a bug that surfaces much later. */
static int read_exact(sema_t *S, const char *path, uint8_t *buf, size_t want,
                      int line) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fail(S, line, "cannot open source file '%s'", path);
        return -1;
    }
    size_t got = fread(buf, 1, want, f);
    int extra = (got == want) && (fgetc(f) != EOF);
    fclose(f);

    if (got != want) {
        fail(S, line, "source file '%s' is %zu bytes but the tensor needs %zu",
             path, got, want);
        return -1;
    }
    if (extra) {
        fail(S, line, "source file '%s' is larger than the %zu bytes the tensor needs",
             path, want);
        return -1;
    }
    return 0;
}

static void handle_tensor(sema_t *S, const nano_stmt_t *st, const nano_env_t *env) {
    nano_program_t *p = S->prog;

    char name[NANOW_MAX_NAME_LEN + 16];
    if (expand_name(S, st->name, env, name, sizeof(name), st->line) != 0) return;
    if (name[0] == '\0') {
        fail(S, st->line, "tensor name must not be empty");
        return;
    }
    if (strlen(name) >= NANOW_MAX_NAME_LEN) {
        fail(S, st->line,
             "tensor name '%s' is %zu bytes; the container stores at most %d",
             name, strlen(name), NANOW_MAX_NAME_LEN - 1);
        return;
    }
    if (find_tensor(p, name)) {
        fail(S, st->line, "duplicate tensor name '%s'", name);
        return;
    }

    check_fields(S, st, TENSOR_KEYS,
                 (int)(sizeof(TENSOR_KEYS) / sizeof(*TENSOR_KEYS)));

    nano_tensor_t t;
    memset(&t, 0, sizeof(t));
    /* The length check above already rejected anything >= NANOW_MAX_NAME_LEN;
     * the precision keeps the compiler able to prove the copy is bounded. */
    snprintf(t.name, sizeof(t.name), "%.*s", (int)sizeof(t.name) - 1, name);
    t.line = st->line;
    t.scale = 1.0f;

    /* --- dtype --- */
    const nano_field_t *fd = field_find(st, "dtype");
    if (!fd) {
        fail(S, st->line, "tensor '%s' is missing required field 'dtype'", name);
        return;
    }
    if (fd->kind != NANO_VAL_IDENT) {
        fail(S, fd->line, "field 'dtype' must be an identifier (ternary, int8, fp16 or fp32)");
        return;
    }
    if (dtype_from_name(fd->str, &t.dtype) != 0) {
        fail(S, fd->line, "unknown dtype '%s' (expected ternary, int8, fp16 or fp32)", fd->str);
        return;
    }
    /* The format defines int8 and fp16, but load_nanow() in run_smollm2.py only
     * decodes ternary and fp32 and skips anything else with a warning. The C
     * reader is dtype-agnostic, so this is about the Python runtime. */
    if (t.dtype == NANOW_DTYPE_FP16 || t.dtype == NANOW_DTYPE_INT8) {
        if (unreadable(S, fd->line,
                       "dtype '%s' is valid in the container format, but the Python "
                       "runtime (run_smollm2.load_nanow) decodes only ternary and "
                       "fp32 and skips this tensor with a warning", fd->str) != 0) {
            return;
        }
    }

    /* --- packing (constrained to what the runtime can read) --- */
    const nano_field_t *fp = field_find(st, "packing");
    if (t.dtype == NANOW_DTYPE_TERNARY) {
        t.packing = NANOW_PACKING_2BIT;
        if (fp) {
            if (fp->kind != NANO_VAL_IDENT) {
                fail(S, fp->line, "field 'packing' must be an identifier");
                return;
            }
            if (strcmp(fp->str, "2bit") == 0) {
                t.packing = NANOW_PACKING_2BIT;
            } else if (strcmp(fp->str, "radix3") == 0) {
                if (unreadable(S, fp->line,
                               "packing 'radix3' has a packer (pack_ternary_radix3) "
                               "but no unpacker: no reader in this repository can "
                               "decode it") != 0) {
                    return;
                }
                t.packing = NANOW_PACKING_RADIX3;
            } else if (strcmp(fp->str, "raw") == 0) {
                if (unreadable(S, fp->line,
                               "packing 'raw' stores one ternary value per byte; the "
                               "runtime expects 2-bit packed ternary") != 0) {
                    return;
                }
                t.packing = NANOW_PACKING_RAW_INT8;
            } else {
                fail(S, fp->line, "unknown packing '%s' (expected 2bit, radix3 or raw)", fp->str);
                return;
            }
        }
    } else {
        t.packing = NANOW_PACKING_RAW_INT8;
        if (fp && !(fp->kind == NANO_VAL_IDENT && strcmp(fp->str, "raw") == 0)) {
            fail(S, fp->line, "packing is only configurable for ternary tensors");
            return;
        }
    }

    /* --- shape --- */
    const nano_field_t *fs = field_find(st, "shape");
    if (!fs) {
        fail(S, st->line, "tensor '%s' is missing required field 'shape'", name);
        return;
    }
    if (fs->kind != NANO_VAL_SHAPE) {
        fail(S, fs->line, "field 'shape' must be a dimension list, e.g. [512, 512]");
        return;
    }
    uint64_t elems = 1;
    t.ndims = (uint32_t)fs->n_shape;
    for (int i = 0; i < fs->n_shape; ++i) {
        if (fs->shape[i] <= 0) {
            fail(S, fs->line, "dimension %d of '%s' must be greater than zero", i, name);
            return;
        }
        if ((uint32_t)fs->shape[i] > NANO_MAX_DIM_VAL) {
            fail(S, fs->line, "dimension %d of '%s' is %d; the supported maximum is %u",
                 i, name, fs->shape[i], NANO_MAX_DIM_VAL);
            return;
        }
        t.shape[i] = (uint32_t)fs->shape[i];
        elems *= (uint64_t)fs->shape[i];
    }
    t.num_elements = elems;

    /* --- scale --- */
    int have_scale = 0;
    if (get_float_field(S, st, "scale", 0, 1.0f, &t.scale, &have_scale) != 0)
        return;
    if (t.dtype == NANOW_DTYPE_TERNARY && !have_scale) {
        fail(S, st->line,
             "ternary tensor '%s' needs an explicit 'scale' (the dequantisation step)",
             name);
        return;
    }
    if (have_scale && !(t.scale > 0.0f)) {
        fail(S, st->line, "scale of '%s' must be greater than zero", name);
        return;
    }

    /* --- init --- */
    const nano_field_t *fi = field_find(st, "init");
    if (!fi) {
        fail(S, st->line, "tensor '%s' is missing required field 'init'", name);
        return;
    }
    if (fi->kind != NANO_VAL_IDENT) {
        fail(S, fi->line, "field 'init' must be an identifier");
        return;
    }
    init_kind_t init = INIT_NONE;
    if (init_from_name(fi->str, &init) != 0) {
        fail(S, fi->line, "unknown init '%s' (expected zeros, ones, ramp, ternary or from)", fi->str);
        return;
    }

    if (t.dtype == NANOW_DTYPE_TERNARY) {
        if (init != INIT_TERNARY && init != INIT_FROM) {
            fail(S, fi->line,
                 "ternary tensor '%s' cannot use init '%s' (expected ternary or from)",
                 name, fi->str);
            return;
        }
    } else if (init == INIT_TERNARY) {
        fail(S, fi->line, "init 'ternary' requires dtype ternary, but '%s' is not", name);
        return;
    }
    if (init == INIT_RAMP && t.dtype != NANOW_DTYPE_FP32 &&
        t.dtype != NANOW_DTYPE_FP16) {
        fail(S, fi->line, "init 'ramp' is only meaningful for fp32 or fp16 tensors");
        return;
    }

    /* --- source (init = from) --- */
    const nano_field_t *fsrc = field_find(st, "source");
    if (init == INIT_FROM) {
        if (!fsrc) {
            fail(S, st->line, "init 'from' on '%s' requires a 'source' path", name);
            return;
        }
        if (fsrc->kind != NANO_VAL_STRING) {
            fail(S, fsrc->line, "field 'source' must be a quoted path");
            return;
        }
    } else if (fsrc) {
        fail(S, fsrc->line, "field 'source' is only valid with init = from");
        return;
    }

    /* --- seed (init = ternary) --- */
    const nano_field_t *fseed = field_find(st, "seed");
    int explicit_seed = 0;
    int seed = 0;
    if (fseed) {
        if (init != INIT_TERNARY) {
            fail(S, fseed->line, "field 'seed' is only valid with init = ternary");
            return;
        }
        if (field_as_int(S, fseed, "seed", &seed) != 0) return;
        explicit_seed = 1;
    }

    /* --- payload --- */
    t.data_size = nano_payload_bytes(t.dtype, t.packing, t.num_elements);
    if (t.data_size == 0) {
        fail(S, st->line, "tensor '%s' has an unsupported dtype/packing combination", name);
        return;
    }

    uint64_t new_total = S->payload_bytes + (uint64_t)t.data_size;
    if (new_total > S->opt->max_total_bytes) {
        fail(S, st->line,
             "tensor '%s' would bring the payload total to %llu bytes, over the "
             "limit of %llu; raise it with --max-bytes if this is intentional",
             name, (unsigned long long)new_total,
             (unsigned long long)S->opt->max_total_bytes);
        return;
    }
    S->payload_bytes = new_total;

    t.payload = (uint8_t *)malloc(t.data_size ? t.data_size : 1);
    if (!t.payload) {
        fail(S, st->line, "out of memory allocating %zu bytes for tensor '%s'",
             t.data_size, name);
        return;
    }

    uint64_t n = t.num_elements;
    switch (init) {
        case INIT_ZEROS:
            memset(t.payload, 0, t.data_size);
            break;

        case INIT_ONES:
            if (t.dtype == NANOW_DTYPE_FP32) {
                float *p32 = (float *)t.payload;
                for (uint64_t i = 0; i < n; ++i) p32[i] = 1.0f;
            } else if (t.dtype == NANOW_DTYPE_FP16) {
                uint16_t h = f32_to_f16(1.0f);
                for (uint64_t i = 0; i < n; ++i) ((uint16_t *)t.payload)[i] = h;
            } else {
                memset(t.payload, 1, t.data_size);
            }
            break;

        case INIT_RAMP:
            if (t.dtype == NANOW_DTYPE_FP32) {
                float *p32 = (float *)t.payload;
                for (uint64_t i = 0; i < n; ++i)
                    p32[i] = (n > 1) ? (float)i / (float)(n - 1) : 0.0f;
            } else {
                uint16_t *p16 = (uint16_t *)t.payload;
                for (uint64_t i = 0; i < n; ++i)
                    p16[i] = f32_to_f16((n > 1) ? (float)i / (float)(n - 1) : 0.0f);
            }
            break;

        case INIT_TERNARY: {
            /* A deterministic LCG: reproducible across machines and runs, which
             * random() is not. Seeded from the tensor name unless the source
             * asked for an explicit seed, so identical declarations in
             * different loop iterations still yield different weights.
             * Documented as a generator, not a source of cryptographic
             * randomness. */
            uint32_t base = explicit_seed ? (uint32_t)seed : name_hash(t.name);
            uint32_t state = base * 2654435761u + 1u;
            int8_t *tmp = (int8_t *)malloc((size_t)n);
            if (!tmp) {
                fail(S, st->line, "out of memory generating tensor '%s'", name);
                free(t.payload);
                t.payload = NULL;
                return;
            }
            for (uint64_t i = 0; i < n; ++i) {
                state = state * 1103515245u + 12345u;
                tmp[i] = (int8_t)((state >> 16) % 3u) - 1;
            }
            size_t written = 0;
            if (t.packing == NANOW_PACKING_2BIT) {
                written = pack_ternary_2bit(t.payload, tmp, (size_t)n);
            } else if (t.packing == NANOW_PACKING_RADIX3) {
                written = pack_ternary_radix3(t.payload, tmp, (size_t)n);
            } else {                       /* raw: one int8 per value */
                memcpy(t.payload, tmp, (size_t)n);
                written = (size_t)n;
            }
            free(tmp);
            if (written != t.data_size) {
                fail(S, st->line,
                     "internal error: packed %zu bytes but the descriptor expects %zu",
                     written, t.data_size);
                free(t.payload);
                t.payload = NULL;
                return;
            }
            break;
        }

        case INIT_FROM: {
            char path[NANO_PATH_CAP];
            if (resolve_path(S, fsrc->str, path, sizeof(path), fsrc->line) != 0) {
                free(t.payload);
                t.payload = NULL;
                return;
            }
            if (read_exact(S, path, t.payload, t.data_size, fsrc->line) != 0) {
                free(t.payload);
                t.payload = NULL;
                return;
            }
            break;
        }

        default:
            break;
    }

    if (push_tensor(S, &t) != 0) {
        free(t.payload);
        return;
    }
    p->total_params += t.num_elements;
}

/* ------------------------------------------------------------------------- */
/* Op block -> nano-ir graph                                                 */
/* ------------------------------------------------------------------------- */

static const char *const OP_KEYS[] = { "in", "out", "scale", "weight" };

static int opcode_from_name(const char *s, nano_ir_opcode_t *op) {
    if (strcmp(s, "embedding_lookup") == 0) { *op = NANO_OP_EMBEDDING_LOOKUP; return 0; }
    if (strcmp(s, "rmsnorm") == 0)          { *op = NANO_OP_RMSNORM;          return 0; }
    if (strcmp(s, "ternary_linear") == 0)   { *op = NANO_OP_TERNARY_LINEAR;   return 0; }
    if (strcmp(s, "ssm_step") == 0)         { *op = NANO_OP_SSM_STEP;         return 0; }
    if (strcmp(s, "silu_mul") == 0)         { *op = NANO_OP_SILU_MUL;         return 0; }
    if (strcmp(s, "residual_add") == 0)     { *op = NANO_OP_RESIDUAL_ADD;     return 0; }
    return -1;
}

static int push_pending_op(sema_t *S, const nano_ir_node_t *node) {
    if (S->n_pending_ops >= S->pending_cap) {
        int cap = S->pending_cap ? S->pending_cap * 2 : 16;
        nano_ir_node_t *grown = (nano_ir_node_t *)realloc(
            S->pending_ops, sizeof(nano_ir_node_t) * (size_t)cap);
        if (!grown) {
            fail(S, 0, "out of memory while buffering IR nodes");
            return -1;
        }
        S->pending_ops = grown;
        S->pending_cap = cap;
    }
    S->pending_ops[S->n_pending_ops++] = *node;
    return 0;
}

static void handle_op(sema_t *S, const nano_stmt_t *st, const nano_env_t *env) {
    char name[NANOW_MAX_NAME_LEN + 16];
    if (expand_name(S, st->name, env, name, sizeof(name), st->line) != 0) return;

    check_fields(S, st, OP_KEYS, (int)(sizeof(OP_KEYS) / sizeof(*OP_KEYS)));

    nano_ir_opcode_t opcode;
    if (opcode_from_name(st->opcode, &opcode) != 0) {
        fail(S, st->line,
             "unknown opcode '%s' (expected embedding_lookup, rmsnorm, "
             "ternary_linear, ssm_step, silu_mul or residual_add)", st->opcode);
        return;
    }

    int dim_in = 0, dim_out = 0;
    if (get_int_field(S, st, "in", 1, 0, &dim_in) != 0) return;
    if (get_int_field(S, st, "out", 1, 0, &dim_out) != 0) return;
    if (dim_in <= 0 || dim_out <= 0) {
        fail(S, st->line, "op '%s' needs positive 'in' and 'out' dimensions", name);
        return;
    }

    float scale = 1.0f;
    if (get_float_field(S, st, "scale", 0, 1.0f, &scale, NULL) != 0) return;

    /* A ternary linear must point at a real ternary weight tensor. This is the
     * link that makes the graph more than documentation: the compiler proves
     * the op and the data agree before writing anything. */
    if (opcode == NANO_OP_TERNARY_LINEAR) {
        const nano_field_t *fw = field_find(st, "weight");
        if (!fw) {
            fail(S, st->line,
                 "op ternary_linear '%s' requires a 'weight' field naming its "
                 "ternary weight tensor", name);
            return;
        }
        if (fw->kind != NANO_VAL_STRING && fw->kind != NANO_VAL_IDENT) {
            fail(S, fw->line, "field 'weight' must be a tensor name");
            return;
        }
        char wname[NANOW_MAX_NAME_LEN + 16];
        if (expand_name(S, fw->str, env, wname, sizeof(wname), fw->line) != 0) return;

        nano_tensor_t *wt = find_tensor(S->prog, wname);
        if (!wt) {
            fail(S, fw->line, "op '%s' names weight '%s', which is not a declared tensor",
                 name, wname);
            return;
        }
        if (wt->dtype != NANOW_DTYPE_TERNARY) {
            fail(S, fw->line, "op '%s' needs a ternary weight, but '%s' is not ternary",
                 name, wname);
            return;
        }
        if (wt->ndims != 2) {
            fail(S, fw->line, "weight '%s' of op '%s' must be 2-D ([out, in])",
                 wname, name);
            return;
        }
        if (wt->shape[0] != (uint32_t)dim_out || wt->shape[1] != (uint32_t)dim_in) {
            fail(S, fw->line,
                 "weight '%s' is [%u, %u] but op '%s' declares out=%d in=%d",
                 wname, wt->shape[0], wt->shape[1], name, dim_out, dim_in);
            return;
        }
    } else if (field_find(st, "weight")) {
        fail(S, st->line, "only ternary_linear accepts a 'weight' field");
        return;
    }

    nano_ir_node_t node;
    memset(&node, 0, sizeof(node));
    node.opcode = opcode;
    if (strlen(name) >= sizeof(node.name)) {
        fail(S, st->line,
             "op name '%s' is %zu characters; the IR node stores at most %zu",
             name, strlen(name), sizeof(node.name) - 1);
        return;
    }
    snprintf(node.name, sizeof(node.name), "%.*s", (int)sizeof(node.name) - 1,
             name);
    node.input_ids[0] = -1;
    node.input_ids[1] = -1;
    node.input_ids[2] = -1;
    node.output_id = S->n_pending_ops;
    node.dim_in = dim_in;
    node.dim_out = dim_out;
    node.scale = scale;
    node.is_ternary = (opcode == NANO_OP_TERNARY_LINEAR);

    push_pending_op(S, &node);
}

/* ------------------------------------------------------------------------- */
/* Statements and loops                                                      */
/* ------------------------------------------------------------------------- */

static void handle_emit(sema_t *S, const nano_stmt_t *st, const nano_env_t *env) {
    char path[NANO_PATH_CAP];
    if (expand_name(S, st->name, env, path, sizeof(path), st->line) != 0) return;
    if (path[0] == '\0') {
        fail(S, st->line, "emit path must not be empty");
        return;
    }
    if (strlen(path) >= sizeof(S->prog->emit_path)) {
        fail(S, st->line, "emit path is too long (limit %zu characters): '%s'",
             sizeof(S->prog->emit_path) - 1, path);
        return;
    }
    snprintf(S->prog->emit_path, sizeof(S->prog->emit_path), "%.*s",
             (int)sizeof(S->prog->emit_path) - 1, path);
    S->prog->have_emit = 1;
    S->n_emit++;
}

static void walk(sema_t *S, const nano_stmt_t *stmts, int n, nano_env_t *env) {
    for (int i = 0; i < n; ++i) {
        const nano_stmt_t *st = &stmts[i];
        switch (st->kind) {
            case NANO_ST_MODEL:
                handle_model(S, st);
                break;

            case NANO_ST_TENSOR:
                handle_tensor(S, st, env);
                break;

            case NANO_ST_OP:
                handle_op(S, st, env);
                break;

            case NANO_ST_EMIT:
                handle_emit(S, st, env);
                break;

            case NANO_ST_FOR:
                if (st->hi < st->lo) {
                    fail(S, st->line,
                         "loop '%s' has an empty range (%d..%d); the end must not "
                         "be less than the start", st->var, st->lo, st->hi);
                    break;
                }
                if (st->hi - st->lo > 65536) {
                    fail(S, st->line, "loop '%s' would unroll %d times; the limit is 65536",
                         st->var, st->hi - st->lo);
                    break;
                }
                if (env->n >= NANO_MAX_ENV) {
                    fail(S, st->line, "loops nested more than %d deep", NANO_MAX_ENV);
                    break;
                }
                for (int v = st->lo; v < st->hi; ++v) {
                    env->slots[env->n].name = st->var;
                    env->slots[env->n].value = v;
                    env->n++;
                    walk(S, st->body, st->n_body, env);
                    env->n--;
                }
                break;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Entry point                                                               */
/* ------------------------------------------------------------------------- */

int nano_sema_run(const nano_ast_t *ast, const nano_options_t *opt,
                  nano_program_t *prog, nano_diag_t *diag) {
    sema_t S;
    memset(&S, 0, sizeof(S));
    S.diag = diag;
    S.opt = opt;
    S.prog = prog;

    /* `from "..."` paths resolve against --from-dir, else the source file's
     * directory (the driver sets source_path before calling us). */
    if (opt->from_dir && opt->from_dir[0]) {
        snprintf(S.base_dir, sizeof(S.base_dir), "%s", opt->from_dir);
    } else if (prog->source_path[0]) {
        snprintf(S.base_dir, sizeof(S.base_dir), "%s", prog->source_path);
        char *slash = strrchr(S.base_dir, '/');
        if (slash) *slash = '\0';
        else S.base_dir[0] = '\0';
    }

    nano_env_t env;
    memset(&env, 0, sizeof(env));

    walk(&S, ast->stmts, ast->n_stmts, &env);

    if (!S.has_model) {
        nano_diag_add(diag, 0, "no model block found; a NanoLang file must declare one");
        S.failed = 1;
    }
    if (prog->n_tensors == 0) {
        nano_diag_add(diag, 0, "no tensors were declared; nothing would be written");
        S.failed = 1;
    }
    if (S.n_emit > 1) {
        nano_diag_add(diag, 0,
                      "%d emit statements found; the last one wins", S.n_emit);
    }

    /* Lower the buffered ops now that the model dimensions are known. */
    if (S.n_pending_ops > 0 && !S.failed) {
        if (nano_ir_graph_init(&prog->ir, S.n_pending_ops, (int)prog->hidden_dim,
                               (int)prog->state_dim, (int)prog->num_layers) != 0) {
            nano_diag_add(diag, 0, "could not allocate the nano-ir graph");
            S.failed = 1;
        } else {
            for (int i = 0; i < S.n_pending_ops; ++i) {
                nano_ir_add_node(&prog->ir, S.pending_ops[i]);
            }
            char err[256] = {0};
            if (!nano_ir_validate_graph(&prog->ir, err, sizeof(err))) {
                nano_diag_add(diag, 0, "nano-ir validation failed: %s", err);
                S.failed = 1;
            }
            prog->ir_enabled = 1;
        }
    }

    free(S.pending_ops);
    return S.failed ? -1 : 0;
}
