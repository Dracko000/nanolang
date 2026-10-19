#!/usr/bin/env bash
#
# run_examples.sh — compile every example and read the result back.
#
# Run with `make -C compiler test`, or directly:
#   NANOC_BIN=/path/to/nanoc compiler/tests/run_examples.sh
#
# This covers what the C unit tests cannot: the shipped command line, the
# page-aligned layout of a real file, byte-for-byte reproducibility across two
# compiles, and — when numpy and the repo's Python runner are present — whether
# the existing runtime decodes every tensor the examples declare.

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"                 # compiler/
repo="$(cd "$root/.." && pwd)"                 # repository root
NANOC="${NANOC_BIN:-$root/build/nanoc}"

if [ ! -x "$NANOC" ]; then
    echo "run_examples: nanoc not found or not executable at $NANOC" >&2
    echo "             build it first with: make -C compiler" >&2
    exit 2
fi

work="$(mktemp -d)"
cleanup() { rm -rf "$work"; }
trap cleanup EXIT

pass=0
fail=0
ok()    { pass=$((pass + 1)); printf '  ok    %s\n' "$1"; }
bad()   { fail=$((fail + 1)); printf '  FAIL  %s\n' "$1"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "run_examples: $NANOC"

# Example name -> expected number of tensors in the container. Hard-coded so a
# regression that silently drops declarations is caught.
declare -A expected_tensors=( ["tiny_lm"]=9 ["demo_ssm"]=10 )

for src in "$root"/examples/*.nano; do
    name="$(basename "$src" .nano)"
    out="$work/$name.nanow"
    ir="$work/$name.nanoir"

    if "$NANOC" "$src" -o "$out" --print-manifest --ir-out "$ir" \
            >"$work/$name.stdout" 2>"$work/$name.stderr"; then
        ok "$name: compiled"
    else
        bad "$name: compile failed"
        sed 's/^/        /' "$work/$name.stderr"
        continue
    fi

    check "$name: container written"          "[ -s '$out' ]"
    check "$name: IR text written"            "[ -s '$ir' ]"
    # 0x4E414E4F little-endian on disk is the bytes 'O','N','A','N'.
    check "$name: begins with the NANOW magic" \
          "[ \"\$(head -c 4 '$out' | od -An -tx1 | tr -d ' \n')\" = '4f4e414e' ]"
    # 64-byte header + 120-byte descriptors, both padded to 4096.
    check "$name: size is page aligned"       "[ \$(( \$(stat -c%s '$out') % 4096 )) -eq 0 ]"
    check "$name: manifest names every tensor" \
          "[ \"\$(awk '/^tensors:/{print \$2}' '$work/$name.stderr')\" = '${expected_tensors[$name]}' ]"

    "$NANOC" "$src" -o "$work/$name.2.nanow" >/dev/null 2>&1
    check "$name: reproducible across compiles" "cmp -s '$out' '$work/$name.2.nanow'"
done

# The CLI's error paths, at the level a user meets them.
printf 'model m { layers = 1 hidden = 4 }\ntensor t { dtype = fp16 shape = [4] init = ones }\n' \
    > "$work/gated.nano"
if "$NANOC" "$work/gated.nano" -o "$work/gated.nanow" >/dev/null 2>&1; then
    bad "fp16 is refused without --allow-unreadable"
else
    ok "fp16 is refused without --allow-unreadable"
fi
if "$NANOC" "$work/gated.nano" -o "$work/gated.nanow" --allow-unreadable >/dev/null 2>&1; then
    ok "fp16 builds with --allow-unreadable"
else
    bad "fp16 builds with --allow-unreadable"
fi
if "$NANOC" --help >/dev/null 2>&1; then ok "--help exits 0"; else bad "--help exits 0"; fi
if "$NANOC" >/dev/null 2>&1; then bad "no arguments is a usage error"; else
    ok "no arguments is a usage error"
fi

# Decode the containers with the repo's own runtime reader.
if command -v python3 >/dev/null 2>&1 && [ -f "$repo/run_smollm2.py" ] \
        && python3 -c 'import numpy' >/dev/null 2>&1; then
    expect_args=()
    for key in "${!expected_tensors[@]}"; do
        expect_args+=("$key=${expected_tensors[$key]}")
    done
    if python3 "$here/verify_nanow.py" "$work" "${expect_args[@]}" \
            >"$work/python.log" 2>&1; then
        ok "python runtime decodes every tensor"
        sed 's/^/        /' "$work/python.log"
    else
        bad "python runtime decodes every tensor"
        sed 's/^/        /' "$work/python.log"
    fi
else
    echo "  skip  python runtime check (needs python3 + numpy + run_smollm2.py)"
fi

printf '\nrun_examples: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
