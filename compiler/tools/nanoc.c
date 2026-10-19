/*
 * nanoc.c — the NanoLang compiler command line.
 *
 *   nanoc <source.nano> [-o out.nanow] [--dump-ir] [--ir-out FILE]
 *                       [--print-manifest] [--from-dir DIR] [--max-bytes N]
 *
 * Human-facing output goes to stderr so that `nanoc model.nano --dump-ir > ir.txt`
 * captures only the graph. See compiler/README.md for the language itself.
 */

#include "nano_lang.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef NANOC_VERSION
#define NANOC_VERSION "0.1.0"
#endif

#define NANOC_EXIT_OK      0
#define NANOC_EXIT_SOURCE  1
#define NANOC_EXIT_USAGE   2
#define NANOC_EXIT_WRITE   3

static void usage(FILE *out, const char *argv0) {
    fprintf(out,
        "nanoc %s — NanoLang compiler\n"
        "\n"
        "usage: %s <source.nano> [options]\n"
        "\n"
        "options:\n"
        "  -o, --output PATH     write the .nanow container to PATH\n"
        "                        (default: `emit` in the source, else <source>.nanow)\n"
        "      --dump-ir         print the nano-ir graph to stdout\n"
        "      --ir-out PATH     also write the nano-ir graph to PATH\n"
        "      --print-manifest  print the compiled tensor table to stderr\n"
        "      --from-dir DIR    base directory for `from \"...\"` sources\n"
        "      --max-bytes N     payload allocation guard (default 536870912)\n"
        "      --allow-unreadable\n"
        "                        downgrade \"no reader can decode this\" from an\n"
        "                        error to a warning (fp16/int8 dtypes, radix3/raw\n"
        "                        ternary packing)\n"
        "  -h, --help            show this message\n"
        "  -v, --version         show the compiler version\n",
        NANOC_VERSION, argv0);
}

/* "dir/model.nano" -> "dir/model.nanow"; anything else gets ".nanow" appended. */
static const char *derive_output(const char *src, char *buf, size_t cap) {
    const char *dot = strrchr(src, '.');
    if (dot && strcmp(dot, ".nano") == 0) {
        size_t stem = (size_t)(dot - src);
        if (stem + sizeof(".nanow") <= cap) {
            memcpy(buf, src, stem);
            memcpy(buf + stem, ".nanow", sizeof(".nanow"));
            return buf;
        }
        return NULL;
    }
    if (strlen(src) + sizeof(".nanow") <= cap) {
        snprintf(buf, cap, "%s.nanow", src);
        return buf;
    }
    return NULL;
}

int main(int argc, char **argv) {
    nano_options_t opt;
    nano_options_default(&opt);

    const char *source = NULL;
    const char *output = NULL;
    int dump_ir = 0;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];

        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(stdout, argv[0]);
            return NANOC_EXIT_OK;
        }
        if (strcmp(a, "-v") == 0 || strcmp(a, "--version") == 0) {
            printf("nanoc %s\n", NANOC_VERSION);
            return NANOC_EXIT_OK;
        }
        if (strcmp(a, "-o") == 0 || strcmp(a, "--output") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "nanoc: %s needs a path\n", a);
                return NANOC_EXIT_USAGE;
            }
            output = argv[i];
            continue;
        }
        if (strcmp(a, "--ir-out") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "nanoc: --ir-out needs a path\n");
                return NANOC_EXIT_USAGE;
            }
            opt.ir_out = argv[i];
            continue;
        }
        if (strcmp(a, "--from-dir") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "nanoc: --from-dir needs a directory\n");
                return NANOC_EXIT_USAGE;
            }
            opt.from_dir = argv[i];
            continue;
        }
        if (strcmp(a, "--max-bytes") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "nanoc: --max-bytes needs a number\n");
                return NANOC_EXIT_USAGE;
            }
            char *end = NULL;
            unsigned long long v = strtoull(argv[i], &end, 10);
            if (!end || *end != '\0' || v == 0) {
                fprintf(stderr, "nanoc: --max-bytes expects a positive integer, got '%s'\n",
                        argv[i]);
                return NANOC_EXIT_USAGE;
            }
            opt.max_total_bytes = v;
            continue;
        }
        if (strcmp(a, "--dump-ir") == 0)        { dump_ir = 1; continue; }
        if (strcmp(a, "--print-manifest") == 0) { opt.print_manifest = 1; continue; }
        if (strcmp(a, "--allow-unreadable") == 0) { opt.allow_unreadable = 1; continue; }

        if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "nanoc: unknown option '%s'\n", a);
            usage(stderr, argv[0]);
            return NANOC_EXIT_USAGE;
        }
        if (source) {
            fprintf(stderr, "nanoc: only one source file is accepted (already have '%s')\n",
                    source);
            return NANOC_EXIT_USAGE;
        }
        source = a;
    }

    if (!source) {
        fprintf(stderr, "nanoc: no source file given\n");
        usage(stderr, argv[0]);
        return NANOC_EXIT_USAGE;
    }

    nano_diag_t diag;
    nano_diag_init(&diag);

    nano_program_t prog;
    if (nano_compile_file(source, &opt, &prog, &diag) != 0) {
        nano_diag_print(&diag, source, stderr);
        nano_program_free(&prog);
        return NANOC_EXIT_SOURCE;
    }

    /* Warnings do not stop the build, but they must still be shown, otherwise
     * --allow-unreadable would be silent about what it allowed. */
    if (diag.count > 0) nano_diag_print(&diag, source, stderr);

    if (opt.print_manifest) nano_print_manifest(&prog, stderr);

    if (dump_ir) {
        if (!prog.ir_enabled) {
            fprintf(stderr, "nanoc: no op statements declared, the IR graph is empty\n");
        } else {
            nano_ir_dump_text(&prog.ir);
        }
    }

    if (opt.ir_out && nano_write_ir_text(&prog, opt.ir_out, &diag) != 0) {
        nano_diag_print(&diag, source, stderr);
        nano_program_free(&prog);
        return NANOC_EXIT_WRITE;
    }

    char derived[512];
    const char *out_path = output;
    if (!out_path && prog.have_emit) out_path = prog.emit_path;
    if (!out_path) out_path = derive_output(source, derived, sizeof(derived));
    if (!out_path) {
        fprintf(stderr, "nanoc: could not derive an output path from '%s'; use -o\n", source);
        nano_program_free(&prog);
        return NANOC_EXIT_USAGE;
    }

    if (nano_emit_nanow(&prog, out_path, &diag) != 0) {
        nano_diag_print(&diag, source, stderr);
        nano_program_free(&prog);
        return NANOC_EXIT_WRITE;
    }

    struct stat st;
    unsigned long long written = 0;
    if (stat(out_path, &st) == 0) written = (unsigned long long)st.st_size;

    fprintf(stderr, "nanoc: %s -> %s (%d tensors, %llu params, %llu bytes)\n",
            source, out_path, prog.n_tensors,
            (unsigned long long)prog.total_params, written);

    nano_program_free(&prog);
    return NANOC_EXIT_OK;
}
