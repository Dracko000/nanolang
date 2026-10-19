#!/usr/bin/env python3
"""Decode compiler-produced .nanow containers with the repository's own reader.

Called by compiler/tests/run_examples.sh. The point is independence: the
compiler writes the container and run_smollm2.load_nanow — code that predates
the compiler and knows nothing about it — reads it back.

Checks, per container:
  * the reader emitted no warning (a warning means it silently skipped a tensor)
  * the number of decoded tensors matches what the example declares
  * ternary payloads hold only -1, 0 and +1 and carry a positive scale
  * fp32 payloads are finite

Usage: verify_nanow.py <dir> <name>=<expected-tensor-count> ...

Exits 0 only when every container passes, so the shell script can report it.
"""

import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from run_smollm2 import load_nanow, unpack_ternary_2bit  # noqa: E402


def check(path, want_count):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tensors, info, mm, fd = load_nanow(path)
    try:
        noise = buf.getvalue().strip()
        if noise:
            raise AssertionError(
                "%s: the reader warned while loading, so something was skipped:\n%s"
                % (os.path.basename(path), noise)
            )

        if len(tensors) != want_count:
            raise AssertionError(
                "%s: the reader decoded %d tensors, expected %d"
                % (os.path.basename(path), len(tensors), want_count)
            )

        for name, t in sorted(tensors.items()):
            if t["ternary"]:
                values = unpack_ternary_2bit(t["data"], t["n_elements"], t["shape"])
                stray = np.setdiff1d(np.unique(values), np.array([-1, 0, 1]))
                if stray.size:
                    raise AssertionError(
                        "%s:%s holds values outside {-1,0,1}: %s"
                        % (os.path.basename(path), name, stray.tolist())
                    )
                if not t["scale"] > 0:
                    raise AssertionError(
                        "%s:%s has a non-positive scale (%r)"
                        % (os.path.basename(path), name, t["scale"])
                    )
            elif not np.isfinite(t["data"]).all():
                raise AssertionError(
                    "%s:%s holds non-finite fp32 values" % (os.path.basename(path), name)
                )

        print(
            "  %s: %d tensors decoded; layers=%s hidden=%s state=%s vocab=%s"
            % (
                os.path.basename(path),
                len(tensors),
                info["n_layers"],
                info["hidden"],
                info["state_dim"],
                info["vocab"],
            )
        )
    finally:
        mm.close()
        os.close(fd)


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    work = argv[1]
    failed = 0
    for spec in argv[2:]:
        name, _, count = spec.partition("=")
        if not count:
            print("bad expectation %r, wanted name=count" % spec, file=sys.stderr)
            return 2
        try:
            check(os.path.join(work, name + ".nanow"), int(count))
        except AssertionError as exc:
            print("FAIL %s" % exc, file=sys.stderr)
            failed += 1
        except OSError as exc:
            print("FAIL %s: %s" % (spec, exc), file=sys.stderr)
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
