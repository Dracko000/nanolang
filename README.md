# NanoLang

**Run a 1.58-bit (ternary) 700M LLM on plain CPUs — no PyTorch, no GPU, no BLAS runtime.**

NanoLang is a small inference engine and agent runtime for ternary-weight language
models. It reads a self-describing `.nanow` container, does the whole forward pass
in NumPy plus hand-written C kernels over AVX2, and exposes an online-grounded
tool-calling agent loop on top. The reference model is a 700M-parameter BitNet
b1.58 checkpoint that decodes at **~22–26 tokens/s on 4 CPU cores** while staying
under **~760 MB of RSS**.

```console
$ python3 run_smollm2.py \
    --model models/bitnet700m/bitnet700m.nanow \
    --config models/bitnet700m/config.json \
    --tokenizer models/bitnet700m/tokenizer.json \
    --prompt "The capital of Indonesia is" --tokens 40 --temp 0.0
```

```
The capital of Indonesia is Jakarta, and the capital of the Philippines is Manila.
...
  [Decode: 40 tokens in 1.55s = 25.8 tok/s]
```

---

## What this is, and what it is not

**Is:** an end-to-end, dependency-light runtime for ternary LLMs plus the tool
harness that makes a 700M model useful. Every operation is either NumPy or C you
can read in one sitting. The point is that the framework stops being a black box.

**Is not:** a new pretrained model. The weights are the upstream
[`1bitLLM/bitnet_b1_58-large`](https://huggingface.co/1bitLLM/bitnet_b1_58-large)
checkpoint (MIT). NanoLang's contribution is the container format, the CPU
kernels, the decoding control (constrained tool calls, KV reuse), the agent
loop, and — since the `nanoc` compiler landed — a description language for
building the container. Accuracy is therefore the upstream model's accuracy —
no better, no worse. See [Attribution](#attribution-and-license).

**Is also not:** a language the runtime executes. `nanoc` compiles
`.nano` source to a `.nanow` container; the runner reads the container and never
parses a `.nano` file. See [NanoLang the language](#nanolang-the-language).

---

## Measured performance

Reference hardware — a *deliberately modest* box, so the numbers are reproducible
on laptops:

| | |
| :--- | :--- |
| CPU | AMD EPYC, 4 cores @ 2.0 GHz |
| ISA | `avx avx2 fma bmi1 bmi2 f16c` — **no AVX-512, no VNNI, no AMX** |
| Cache | L1d 128 KB · L2 2 MB · L3 8 MB |
| RAM | 8 GB, no swap, no GPU |
| Threads | `OMP_NUM_THREADS = cpu_count - 1` (3) · `OPENBLAS_NUM_THREADS = 1` |

| Metric | Value |
| :--- | ---: |
| Weights on disk (`.nanow`) | **350.5 MB** |
| Peak RSS while decoding | **~754–760 MB** |
| Decode, short context | **22–26 tok/s** |
| Prefill (18-token prompt) | 33 tok/s |
| Framework dependencies | `numpy` only (+ stdlib, `ctypes`) |

The first working single-threaded version ran at **1.14 tok/s**. It now runs at
**~22–26 tok/s** — a ~20× gain from bit-packing, an int8-activation/int32-accumulate
GEMV, OpenMP across the 4 cores, and folding batch dims away.

> Run-to-run variance on this shared box is ±5–10%. Where a difference was near
> the noise floor it is reported as such rather than rounded up.

### Where the time goes

Each token performs, per layer: **4 RMSNorm** (input, `inner_attn_ln`,
post-attention, `ffn_layernorm`), **2 RoPE**, **1 SiLU-mul**, **1 softmax** —
**193 elementwise ops per token** across 24 layers. Everything else is the
ternary GEMV, which scales `O(hidden²)` and dominates:

| | ms/token | share |
| :--- | ---: | ---: |
| Ternary GEMV + attention | ~36 | ~95% |
| Elementwise (C kernels) | ~2.1 | ~5% |

Because the linear layers are ~95% of the time, kernel work on the *elementwise
ops* only pays off once the GEMV gets faster (NEON, NPU). That is exactly why the
C elementwise path exists and ships enabled — see
[ROADMAP §7.7](ROADMAP_IMPLEMENTASI.md).

A streaming memory probe on this same box reaches **~16 GB/s**, so a 350 MiB
container caps decode at **~41–44 tok/s**. At 22–26 tok/s the runtime therefore
sits at **~51–60 % of that roofline** — it is *not* purely bandwidth-limited. The
2-bit unpack and the int8 add/sub work in the GEMV are the other candidate, and
**which of the two ceilings binds is still an open question**. Answering it is the
prerequisite for deciding whether shrinking the container (see the embedding note
below) actually buys throughput.

---

## The `.nanow` container

One file, memory-mapped, self-describing. No pickle, no GPU tensors, no side files.

```
┌──────────────────────────────────────────────┐
│ header: magic, version, tensor count, config │
├──────────────────────────────────────────────┤
│ tensor directory: name, dtype, shape, offset │
├──────────────────────────────────────────────┤
│ payload: 168 ternary tensors (2-bit packed)  │
│          + fp32 norms / embeddings           │
└──────────────────────────────────────────────┘
```

| Property | Value |
| :--- | :--- |
| Size | 367,513,600 bytes (350.5 MB) for the 700M model |
| Tensors | 266 total — **168 ternary**, the rest fp32 norms/embeddings |
| Ternary encoding | 2 bits/weight, values `{-1, 0, +1}`, one f32 scale per row |
| Activation path | dynamic INT8, INT32 accumulation, single rescale per row |
| Load strategy | `mmap` file → copy into RAM → close the descriptor |

The mmap is closed immediately after load, so the 350 MB file is not held open and
does not inflate RSS beyond the resident weight copy.

Reproduce the container from any HF checkpoint:

```bash
python3 convert_hf_to_nanow.py --hf 1bitLLM/bitnet_b1_58-large \
    --out models/bitnet700m/bitnet700m.nanow
```

---

## NanoLang the language

`.nanow` is the artefact. `.nano` is the source that produces it. `nanoc` is the
compiler, in `compiler/`, and it is what finally makes the "language" half of the
name true.

```text
model.nano  --[nanoc]-->  model.nanow  --[the runner]-->  tokens
```

A `.nano` file declares the model header, every tensor with its dtype, shape,
scale and initialiser, and an optional op graph. It has loops and `${var}`
interpolation, so twenty-four layers are three declarations inside one `for`, not
seventy-two hand-written ones.

```text
model tiny_lm { layers = 2  hidden = 64  vocab = 128 }

for layer in 0..2 {
  tensor "layers.${layer}.w_in" {
    dtype = ternary  shape = [64, 64]  scale = 0.02  init = ternary
  }
  op ternary_linear "layers.${layer}.in_proj" {
    in = 64  out = 64  scale = 0.02  weight = "layers.${layer}.w_in"
  }
}
```

Three properties are worth calling out, because they are the reason this is a
language and not a config file:

- **It checks before it writes.** A `ternary_linear` op must name a weight tensor
  that exists, is ternary, is 2-D, and is shaped `[out, in]`. Loop variables must
  be bound. Shapes must be positive. A `from "file.bin"` payload must be exactly
  the right number of bytes. Nothing is written until every declaration is
  consistent.
- **It is reproducible.** Two compiles of the same source produce byte-identical
  containers, asserted with `cmp` in the test suite.
- **It borrows the runtime's writer.** Codegen calls phase1's `nanow_write_file`,
  and op statements are lowered into phase1's `nano_ir` graph and validated by
  `nano_ir_validate_graph`. The container format therefore still has exactly one
  implementation. The IR is a compile-time check and an inspection dump
  (`--dump-ir`); it is not stored in the container and the runner does not read it.

### Extensions

Real extensions for code, `.nanow` for artefacts of this format.

| File | What it is |
| :--- | :--- |
| `*.nano` | NanoLang source |
| `*.nanow` | the binary container (weights + directory) |
| `*.nanoir` | optional IR text dump from `--ir-out` |
| `*.c` `*.h` `*.py` | the compiler, the kernels, the runner |

The C sources stay `.c`/`.h` deliberately. `gcc -c x.nanow` silently produces no
object file because the extension no longer selects the language, so renaming them
to `.nanow` would break the build in a way that looks like a missing file.

Full specification, grammar, field reference and the list of what the compiler
rejects: [`compiler/README.md`](compiler/README.md).

```bash
make -C compiler
compiler/build/nanoc compiler/examples/tiny_lm.nano -o /tmp/tiny_lm.nanow --print-manifest
make -C compiler test      # 188 assertions + 19 end-to-end checks
```

---

## Feature map

| Feature | Flag | What it does |
| :--- | :--- | :--- |
| Ternary GEMV | *(default)* | INT8 activations × 2-bit weights → INT32 accumulate, AVX2 + OpenMP |
| C elementwise kernels | `--ew {auto,all,off}` | RMSNorm / RoPE / SiLU / softmax in C, each 1.9×–2.6× NumPy |
| Online grounding | `--online --ctx-chars N` | Wikipedia + DuckDuckGo retrieval, stdlib only |
| Tool calling | `--agent` | 10 sandboxed tools, typed schemas, tolerant parser |
| Constrained decoding | `--constrained-args` | Tool **name** masked at the logit level, **args** forced to schema, values free |
| KV reuse within a run | *(default)* | Token-level `feed()`; a re-encode never matches because SPM decode re-adds a leading space |
| KV reuse across runs | `--kv-cache-dir DIR` | Prefix KV persisted to `.npz`, keyed by `sha1(model_tag + prefix ids)` |
| Multi-step planning | `--planner {model,route,hybrid}` | `route` splits on explicit sequencing connectors (*then*, *after that*) — never a bare *and* |
| Prompt size control | `--prompt-style {auto,full,compact}` | Compact prefix under constrained decoding, full otherwise |
| Loop protection | *(default)* | Identical `(name, sorted args)` twice → stop and return the first result |

### Constrained decoding is what makes tool calls work

A 700M model is not good at emitting well-formed tool calls by free generation:
**5/8** parseable at best with the full prompt, **1/8** with the compact one. So
NanoLang does not rely on free generation for structure:

| Mode | Parsable | Correct tool |
| :--- | ---: | ---: |
| Free generation, full prefix | 5/8 | 5/8 |
| Free generation, compact prefix | 1/8 | 1/8 |
| **Constrained**, full prefix | — | **9/10** |
| **Constrained**, compact prefix | — | **9/10** |

A small model cannot be trusted to *format*; it can be trusted to *choose*
inside a forced alphabet. `--constrained-args` is the recommended mode.

### KV cache: the biggest single win for agent workloads

The agent preamble (tool list + few-shot) is a static prefix. Its KV state is a
pure function of the token ids and the weights, so it is safe to serialize.

| | prefill | wall clock |
| :--- | ---: | ---: |
| No cache | 444 tokens every run | 45.7 s |
| Cold run (fills cache) | 444 tokens + write 131 MB | 56.6 s |
| Warm run (loads cache) | 0 tokens | **13.5 s** |

Then the prefix itself was shrunk. The compact prefix is **172 tokens (−61 %)**:

| | cold | warm | cache |
| :--- | ---: | ---: | ---: |
| Full prefix | 56.6 s | 13.5 s | 131 MB |
| Compact prefix | **18.0 s** | **8.0 s** | **51 MB** |

`--prompt-style auto` picks compact exactly when the decoding is constrained,
which is precisely when it costs nothing in accuracy.

---

## Quickstart

```bash
git clone https://github.com/Dracko000/nanolang && cd nanolang

# 1. Build the C kernels
make                 # -> phase0/bin/libternary.so

# 2. Get the containerised model (see the model repo for the HF link)
mkdir -p models/bitnet700m && cd models/bitnet700m
#   download bitnet700m.nanow, config.json, tokenizer.json here
cd ../..

# 3. Talk to it
python3 run_smollm2.py --model models/bitnet700m/bitnet700m.nanow \
  --config models/bitnet700m/config.json \
  --tokenizer models/bitnet700m/tokenizer.json \
  --prompt "Explain what a ternary weight is, briefly." --tokens 120 --temp 0.7
```

### Online, grounded

```bash
python3 run_smollm2.py --model models/bitnet700m/bitnet700m.nanow \
  --config models/bitnet700m/config.json \
  --tokenizer models/bitnet700m/tokenizer.json \
  --online --ctx-chars 1200 \
  --prompt "What happened in the 2026 Indonesian Grand Prix?" --tokens 150
```

### As an agent

```bash
python3 run_smollm2.py --model models/bitnet700m/bitnet700m.nanow \
  --config models/bitnet700m/config.json \
  --tokenizer models/bitnet700m/tokenizer.json \
  --agent --constrained-args --planner route \
  --workspace ./nanow_workspace --allow-bash \
  --kv-cache-dir ./.kv \
  --prompt "create todo.txt with the content buy milk then read it" \
  --max-steps 3
```

```
  🧭 router -> write_file({"path": "todo.txt", "content": "buy milk"}) -> 'wrote 8 bytes to todo.txt'
  🧭 router -> read_file({"path": "todo.txt"}) -> 'buy milk'
```

The router turns that sentence into `write_file(todo.txt, "buy milk")` followed by
`read_file(todo.txt)` — the pronoun *it* inherits the path from the previous step.

> **Honest caveat:** the *tool calls* are correct; the *prose summary* the model
> writes at the end is not. A 700M base model has no instruction tuning, so its
> closing sentence frequently drifts (a real run ended with `I'm trying to do this
> in C++:`). Trust the tool transcript, not the narration.

---

## Layout

```
compiler/ nanoc: .nano source -> .nanow container (uses phase1's writer + nano-ir)
phase0/   ternary math + AVX2/NEON kernels + C elementwise ops  -> libternary.so
phase1/   .nanow container: writer, mmap loader, nano-ir, SSM step
phase2/   engine + QNN/Metal backends, distilled SSM, CLI
phase3/   semantic DSL bytecode, renderer, iOS/Android bridges
tests/    pipeline + tool-runtime unit tests
run_smollm2.py            the inference runner (tokenizer, attention, decoding, web)
nanow_tools.py            tool registry, sandbox, router, agent loop
convert_hf_to_nanow.py    HF checkpoint  -> .nanow
convert_smollm2.py        SmolLM2        -> .nanow
WHITEPAPER_NANOLANG.md    architecture spec (aspirational; see the caveat inside)
ROADMAP_IMPLEMENTASI.md   engineering log with every measured finding
```

### What actually runs

The directories above are **not all on the same footing**. The released model
exercises `phase0` and the Python runner; the rest is either a buildable side demo
or a proposal that is not wired into anything.

| Component | Where | Used by the released model? |
| :--- | :--- | :--- |
| Ternary GEMV + elementwise C kernels | `phase0` | **Yes** — loaded through ctypes as `libternary.so` |
| `.nanow` reader (header, tensor directory, 2-bit unpack) | `phase1/src/nanow_mmap.c`, mirrored in `run_smollm2.py` | **Yes** — the runner parses the real container |
| Tokenizer, attention, RoPE, sampling, KV cache, web, tools | `run_smollm2.py`, `nanow_tools.py` | **Yes** |
| `.nanow` writer | `phase1/src/nanow_writer.c` | Yes — used by `convert_hf_to_nanow.py` and by `nanoc` |
| `nanoc` compiler, `.nano` language | `compiler/` | **Yes, at build time** — compiles sources to containers; the runner never sees a `.nano` file |
| `nano-ir` (MLIR-style dialect) | `phase1/src/nano_ir.c` | Yes, at compile time — `nanoc` lowers `op` statements into the graph and runs `nano_ir_validate_graph` on it. Not stored in the container, not read by the runner |
| Ternary-SSM step | `phase1/src/ssm_step.c` | **No** — the released model is attention-based and has 0 SSM tensors |
| Engine CLI, QNN and Metal backends, SSM distillation | `phase2` | **No** — builds and runs as its own benchmark |
| Semantic DSL bytecode, renderer, iOS/Android bridges | `phase3` | **No** — builds and runs as its own demo |

Put plainly: **the `.nanow` container format is real and in production, and the
`.nano` language is real as a build-time description of that container. What the
runtime executes is still the container alone.** "NanoLang" as a semantic-bytecode
language on the inference path is still not a thing; `nano-ir` is a compiler pass,
not a VM. The whitepaper describes three layers; what you get from the quickstart
is the container from Layer 1, the ternary arithmetic of Layer 2, and the compiler
that writes the container.

### Tests

```bash
make -C phase0 lib
make -C compiler test                        # 188 assertions + 19 end-to-end checks

python3 -m unittest tests.test_pipeline -v   # 32 tests
python3 -m unittest tests.test_tools -v      # 42 tests
python3 phase0/python/test_bitnet.py         # BitNet math
python3 phase3/python/test_semantic_dsl.py   # semantic DSL
```

The suite covers ternary quantisation and 2-bit packing, INT8/INT32 GEMV against a
reference, container round-trip, RoPE, tokenizer name-mapping, KV persistence
round-trip, C-vs-NumPy parity for every elementwise kernel (including wide-range
`exp` values), prompt-style selection, tool-call parsing, path-sandbox escapes,
the bash deny-list, schema validation, the router, dedup, and the agent loop
against a fake runner.

`make -C compiler test` additionally compiles both shipped examples, asserts the
containers are page-aligned and byte-identical across recompiles, and decodes them
with `run_smollm2.load_nanow` — a reader that knows nothing about the compiler — to
prove the compiler's output is an ordinary `.nanow` file.

---

## Limitations, honestly

- **The model, not the runtime, limits quality.** 700M parameters produce a
  700M-parameter-quality answer. The runtime makes it *fast and grounded*, not
  *smart*.
- **Constrained tool calls are reliable; the prose around them is not.** The call
  the model makes is correct, but the natural-language summary it generates
  afterwards frequently drifts. Read the tool transcript, not the narration.
- **Elementwise C kernels buy ~5 % end-to-end today.** They are 1.9×–2.6× faster
  per op ([best-of-6 decode](ROADMAP_IMPLEMENTASI.md): 43.74 → 37.95 ms/token,
  1.15×), but the GEMV is 95 % of the time. Adopted because they are correct,
  parity-tested, and future-proof — not because they moved the headline number.
- **No AVX-512 / VNNI / AMX path yet**, and no GPU/NPU backend wired to this
  runner. `phase2/` has QNN and Metal sources; they are not exercised by the
  quickstart.
- **`search the web …` is a known soft spot.** Under `--planner model` the model
  picks `http_get` with a fabricated URL. Use `--planner route` for reliability.
- **`--online` retrieves and injects; it does not cite.** Snippets are prepended
  with their source, but the model may still misattribute.
- **Bash is opt-in and deny-listed**, not a true sandbox. Do not enable
  `--allow-bash` on a machine you care about.

---

## Attribution and license

- **Code:** MIT — see [LICENSE](LICENSE).
- **Model weights:** the `.nanow` container redistributes
  [`1bitLLM/bitnet_b1_58-large`](https://huggingface.co/1bitLLM/bitnet_b1_58-large)
  (MIT, a reproduction of *"The Era of 1-bit LLMs: All Large Language Models are
  in 1.58 Bits"*, [arXiv:2402.17764](https://arxiv.org/abs/2402.17764), trained
  on RedPajama for 100B tokens). That project's license and attribution are
  preserved in the model repo.
- NanoLang is an independent project. It is not affiliated with or endorsed by
  the authors of BitNet, Microsoft, or Hugging Face.
