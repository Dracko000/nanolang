# NANOLANG & TERNARY-SSM
## A Tri-Layer Architecture for Zero-MatMul, $O(1)$ Memory Edge Intelligence on Heterogeneous Mobile Silicon

**Document Classification:** Technical Whitepaper & Architectural Specification  
**Version:** 1.0.0-DRAFT  
**Date:** October 2026  
**Status:** Architecture Proposal & Feasibility Blueprint  

---

### Executive Summary

Modern Large Language Models (LLMs) are constrained by severe physical limitations on mobile hardware:
1. **The Memory Bandwidth Wall:** Autoregressive decoding requires transferring billions of model parameters across low-power DRAM (LPDDR5X, ~50–75 GB/s) for every single token generated.
2. **Quadratic KV-Cache Bloat:** Standard Transformer self-attention requires caching Key-Value states that grow linearly with context length $O(N)$ and compute at $O(N^2)$, exhausting mobile RAM in extended dialogues.
3. **Thermal & Energy Constraints:** Mobile form factors enforce a sustained thermal dissipation limit of 3–5 Watts. High-precision Floating-Point Multiply-Accumulate (FP-MAC) operations rapidly trigger thermal throttling and battery depletion.

This paper proposes **NanoLang & the Ternary-SSM Stack**, a co-designed tri-layer computational architecture engineered from first principles for sub-watt, native on-device execution:
* **Layer 1 (Bare-Metal Compiler & Execution Engine):** A deterministic, zero-garbage-collection systems language and MLIR-based intermediate representation (`nano-ir`) targeting mobile Unified Memory Architectures (UMA) with zero-copy memory mapping.
* **Layer 2 (Mathematical & Structural Engine):** A **MatMul-Free Ternary Model ($W \in \{-1, 0, +1\}$)** unified with a **Selective State Space Model (SSM)** backbone. FP-MAC operations are replaced by integer additions and subtractions, and the KV-cache is replaced by a fixed-size $O(1)$ hidden state.
* **Layer 3 (Dense Semantic Bytecode DSL):** A high-entropy symbolic representation layer that replaces verbose natural language token generation with compact Abstract Syntax Tree (AST) primitives, reducing decoding sequence length by 60–75% before local deterministic text reconstruction.

Together, this stack theoretically delivers up to **10–15× reduction in DRAM bandwidth requirements**, **20× reduction in arithmetic energy expenditure**, and guarantees **sustained >30 tokens/second execution at under 2.5 Watts** on contemporary smartphone processors.

---

### Implementation Status

This document is a design proposal. Not all of it is built, and it is worth being
explicit about which parts are, because the shipped system is narrower than the
architecture below. Status as of October 2026:

| Layer | Component | Status |
| :--- | :--- | :--- |
| 2 | Ternary weights `{-1, 0, +1}`, 2-bit packed | **Shipped** — a 700M checkpoint runs from a 350 MiB container |
| 2 | INT8 activation × ternary weight → INT32 accumulate, AVX2 | **Shipped** — measured 22–26 tok/s on 4 CPU cores |
| 1 | `.nanow` container: header, tensor directory, mmap, writer | **Shipped** — this is the format the released model uses |
| 1 | Elementwise kernels (RMSNorm, RoPE, SiLU, softmax) in C | **Shipped** — including a float32-accurate AVX2 `exp` |
| 2 | Ternary **SSM** recurrence replacing attention | **Prototype only** — `phase1/src/ssm_step.c` builds and runs a synthetic demo; the released 700M model is an ordinary attention transformer and contains no SSM tensors |
| 1 | `nano-ir` MLIR dialect and lowering to ANE/Hexagon | **Prototype only** — `phase1/src/nano_ir.c` is a self-contained data-structure demo; no compiler, no MLIR integration, and nothing in the inference path consumes it |
| 1 | Bare-metal systems language, zero-GC runtime | **Not started** — there is no language implementation in this repository |
| 1 | QNN / Metal / Hexagon lowering backends | **Source only** — `phase2/` holds a QNN graph builder and a Metal kernel; neither is exercised by any runnable path |
| 3 | Semantic bytecode DSL and deterministic renderer | **Prototype only** — `phase3/` builds a demo that reports a token-sequence reduction, but the inference runner does not generate or consume semantic primitives |

The projections in Section 5 are analytical estimates for the full three-layer
stack on fixtures that do not all exist. The measured numbers for what *is* built
are in [ROADMAP_IMPLEMENTASI.md](ROADMAP_IMPLEMENTASI.md) and the repository README.

---

```
+-------------------------------------------------------------------------+
|                        LAYER 3: SEMANTIC DSL                            |
|    Natural Language Prompt -> High-Entropy AST -> Dense Bytecode Tokens |
|    (60-75% reduction in token sequence length via symbolic primitives)  |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------v------------------------------------+
|                  LAYER 2: TERNARY-SSM MATHEMATICS                       |
|   * Weights in {-1, 0, +1} (1.58-bit BitNet Representation)             |
|   * Zero Floating-Point Multiplications (Matrix Multiply -> ADD/SUB)    |
|   * Recurrent State-Space Formulation (Zero KV-Cache, O(1) Memory)      |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------v------------------------------------+
|                  LAYER 1: COMPILER & RUNTIME CORE                       |
|   * nano-ir (MLIR dialect) with direct lowering to ANE / Hexagon / NEON |
|   * Unified Memory Architecture (UMA) zero-copy buffer management       |
|   * Direct Flash-to-SRAM paging (No Python, No Garbage Collector)       |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------v------------------------------------+
|                     MOBILE HARDWARE TARGETS                             |
|       Apple Silicon (ANE/Metal) | Qualcomm Snapdragon (Hexagon/Adreno)   |
+-------------------------------------------------------------------------+
```

---

## 1. The Physics of Mobile Inference: Failure of the Standard Stack

### 1.1 The Arithmetic vs. Bandwidth Imbalance
In LLM inference, two distinct computational regimes exist:
* **Prefill Phase (Prompt Processing):** Compute-bound. Tokens are processed in parallel, saturating matrix multiplication units ($O(N)$ operations per parameter).
* **Decode Phase (Token Generation):** Memory-bandwidth-bound. Tokens are generated strictly autoregressively, one by one. For each token, the hardware must stream the *entire* parameter weight matrix from DRAM into on-chip cache:

$$\text{Throughput}_{\text{decode}} \le \frac{\text{Memory Bandwidth (Bytes/s)}}{\text{Model Footprint (Bytes)}} \quad \text{[Tokens/s]}$$

On server-grade hardware (NVIDIA H100 with HBM3e), memory bandwidth reaches **3,350 GB/s**.  
On flagship mobile hardware (Qualcomm Snapdragon 8 Gen 3 / Apple A18 Pro with LPDDR5X), bandwidth is capped at **~50 to 75 GB/s**.

| Metric | Server (H100 SXM) | Mobile Flagship (LPDDR5X) | Deficit Ratio |
| :--- | :--- | :--- | :--- |
| **Peak Memory Bandwidth** | 3,350 GB/s | 64 GB/s | **~52× slower** |
| **Sustained Thermal Budget** | 700 Watts | 3.5 – 5.0 Watts | **~140× smaller** |
| **Unified System Memory** | 80 GB HBM3e | 8 – 16 GB LPDDR5X | **~5–10× smaller** |

### 1.2 The KV-Cache Memory Explosion
Standard Multi-Head Attention (MHA) caches Key and Value activations for every past token. For context length $L$, layers $N_{\text{layers}}$, heads $N_{\text{heads}}$, and head dimension $D$:

$$\text{Memory}_{\text{KVCache}} = 2 \times N_{\text{layers}} \times N_{\text{heads}} \times D \times L \times \text{BytesPerParam}$$

For a 7B parameter model (e.g., Llama-3 8B in FP16) with a modest 8,192 token conversation:
$$\text{Memory}_{\text{KVCache}} = 2 \times 32 \times 32 \times 128 \times 8192 \times 2 \approx 4.29 \text{ GB}$$

Adding 4.29 GB of dynamic KV-cache to a 14 GB model weight footprint instantly exceeds the typical total physical RAM of an iPhone or Android flagship (8–12 GB), triggering operating system memory kills (OOM).

---

## 2. Layer 2: Mathematical Foundation — MatMul-Free Ternary SSM

### 2.1 1.58-bit Ternary Weight Formulation
Following the BitNet b1.58 paradigm, every weight parameter $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ is constrained to the discrete ternary set:

$$\mathcal{T} = \{-1, 0, +1\}$$

#### Weight Quantization Function
During forward execution and post-training distillation, real-valued weights $W$ are quantized using an absolute-mean scaling factor $\gamma$:

$$\gamma = \frac{1}{d_{\text{in}} \cdot d_{\text{out}}} \sum_{i,j} |W_{i,j}|$$

$$\widetilde{W}_{i,j} = \text{Clip}\left(\left\lfloor \frac{W_{i,j}}{\gamma} \right\rceil, -1, 1\right)$$

#### Activation Quantization
Activations $X \in \mathbb{R}^{B \times d_{\text{in}}}$ are dynamically quantized to INT8 with scaling factor $\beta = \max(|X|) / 127$:

$$\widetilde{X} = \text{Clip}\left(\left\lfloor \frac{X}{\beta} \right\rceil, -128, 127\right)$$

### 2.2 Complete Elimination of Floating-Point Multiplications
In conventional linear layers, calculating $Y = X W$ requires $2 \cdot M \cdot N \cdot K$ floating-point MACs.  
Under Ternary weights $\widetilde{W} \in \{-1, 0, +1\}$, the dot product simplifies to purely conditional addition and subtraction:

$$Y_{i,k} = \sum_{j=1}^{d_{\text{in}}} \widetilde{X}_{i,j} \cdot \widetilde{W}_{j,k} = \sum_{j \in \mathcal{P}_k} \widetilde{X}_{i,j} - \sum_{j \in \mathcal{N}_k} \widetilde{X}_{i,j}$$

where:
* $\mathcal{P}_k = \{j \mid \widetilde{W}_{j,k} = +1\}$ (Indices added)
* $\mathcal{N}_k = \{j \mid \widetilde{W}_{j,k} = -1\}$ (Indices subtracted)
* Entries where $\widetilde{W}_{j,k} = 0$ are skipped entirely (*Zero-Skip Acceleration*).

**Energy Advantage:** In modern 3nm/4nm silicon:
* 16-bit FP-MAC: **~1.1 picojoules (pJ)**
* 8-bit Integer Addition: **~0.03 – 0.05 picojoules (pJ)**  
$\implies$ **~22× reduction in arithmetic execution energy.**

### 2.3 Replacing Self-Attention with Selective State Space Models (SSM)
To permanently eliminate the quadratic KV-cache memory bottleneck, NanoLang replaces Transformer Attention blocks with a discretized continuous-time State-Space recurrence (Mamba/S4 style):

$$h_t = \mathbf{\bar{A}}_t h_{t-1} + \mathbf{\bar{B}}_t x_t$$
$$y_t = \mathbf{C}_t h_t + \mathbf{D} x_t$$

Where:
* $h_t \in \mathbb{R}^{D \times N}$ is the recurrent hidden state (typically $N=16$, $D=2048$).
* $\mathbf{\bar{A}}_t = \exp(\Delta_t \mathbf{A})$, where $\Delta_t$ is an input-dependent discretization parameter.

#### Invariant Memory Footprint:
Unlike Transformers where memory increases linearly $O(L)$, the SSM state size is **strictly constant**:

$$\text{Memory}_{\text{State}} = D \times N \times \text{Bytes} = 2048 \times 16 \times 2 = 65,536 \text{ Bytes } (64 \text{ KB})$$

64 Kilobytes fits entirely into the L1/L2 SRAM cache of any mobile NPU core.

---

## 3. Layer 1: The Compiler & Execution Engine (`NanoLang-Core`)

### 3.1 Design Principles
1. **No Runtime Overhead:** Zero garbage collector, zero reference counter, zero dynamic dispatch.
2. **Native Memory Mapping (`UMA-MMap`):** Weights remain on UFS 4.0 storage and are mapped directly into unified virtual address spaces accessible by CPU, GPU, and NPU without intermediary copies.
3. **Bit-Packed Storage:** Because $\log_2(3) \approx 1.58$ bits, ternary values are encoded using 2 bits per weight (with optional radix-3 packing storing 5 ternary values per byte, achieving 1.6 bits/weight).

### 3.2 Intermediate Representation (`nano-ir`)
`nano-ir` is an open dialect built on LLVM/MLIR, structured around ternary tensor expressions:

```mlir
// nano-ir Dialect Example: Ternary Linear Transformation
module {
  func.func @ternary_layer(%act: tensor<?x2048xi8>, 
                           %weight_bits: tensor<2048x4096x2bit>, 
                           %scale_alpha: f32) -> tensor<?x4096xi8> {
    
    // Lowering to hardware-specific integer accumulator vector ops
    %res = "nano.ternary_accumulate"(%act, %weight_bits) {
      skip_zeros = true,
      vector_width = 128 : i32
    } : (tensor<?x2048xi8>, tensor<2048x4096x2bit>) -> tensor<?x4096xi32>
    
    %scaled = "nano.scale_and_clamp"(%res, %scale_alpha) : (tensor<?x4096xi32>, f32) -> tensor<?x4096xi8>
    return %scaled : tensor<?x4096xi8>
  }
}
```

### 3.3 Target Hardware Lowering Backends

```
                        +----------------------+
                        |   nano-ir Bytecode   |
                        +----------+-----------+
                                   |
             +---------------------+---------------------+
             |                                           |
+------------v------------+                 +------------v------------+
|  Apple Silicon Backend  |                 | Qualcomm Hexagon Backend|
|  * ANE MIL lowering     |                 | * Hexagon Vector (HVX)  |
|  * Metal Shading Language|                | * Tensor Processor (HTP)|
|  * MPSGraph INT8 accum  |                 | * QNN SDK Direct Graph  |
+-------------------------+                 +-------------------------+
```

---

## 4. Layer 3: Semantic DSL & Dense Bytecode Tokenization

### 4.1 The Shannon Entropy Problem in Natural Language
Natural language is inherently redundant. When an LLM outputs:
> *"The function takes two arguments and returns their sum."* (11 tokens, ~45 bytes)

It transmits less than 20 bits of true structural semantic information. Every redundant token incurs an entire forward-pass memory sweep.

### 4.2 Dense AST Tokenization
NanoLang defines an intermediate symbolic grammar. The on-device neural model generates compact structural primitives rather than conversational text:

```
[Raw Human Query]
  |
  v (Mobile Input Parsing)
[Semantic Target Intent]
  |
  v (NanoLang On-Device Generator generates 4 Bytecode Tokens)
{ OP: FUNC_DEF, ARGS: 2, RET: ADD(A, B) }
  |
  v (Local Deterministic Template Renderer, 0 FLOPs Neural)
"The function takes two arguments and returns their sum."
```

By delegating syntactic fluff to local deterministic template expansion, the neural engine decodes **only 3–5 dense structural tokens instead of 25 natural language tokens**, yielding an immediate **5× throughput improvement**.

---

## 5. Theoretical Feasibility & Performance Projections

### 5.1 System Parameters Comparison
Evaluating a 3.5B Parameter NanoLang-SSM vs. conventional Llama-3 8B on a smartphone with **LPDDR5X (64 GB/s)**:

| Specification | Llama-3 8B (INT4) | NanoLang-SSM 3.5B (Ternary 1.58b) | Advantage Factor |
| :--- | :--- | :--- | :--- |
| **Model Weight Size** | 4.3 GB | **0.69 GB** (Radix-3 packed) | **6.2× smaller** |
| **KV-Cache (8k tokens)** | 1.1 GB (GQA) | **64 KB** (Fixed SSM State) | **>17,000× smaller** |
| **DRAM Read per Token** | 4.3 GB | **0.69 GB** | **6.2× less bandwidth** |
| **Theoretical Peak Speed** | ~14.8 tokens/sec | **~92.7 tokens/sec** | **6.3× faster** |
| **Arithmetic Type** | INT4-FP16 MAC | **INT8 Addition / Subtraction** | **Eliminates Mul circuits** |
| **Estimated System Power** | 6.5 W (Thermal Throttles) | **1.8 – 2.4 W (Continuous)** | **Sustained battery life** |

---

## 6. Conclusion
The combination of **Ternary Precision ($\{-1, 0, +1\}$)**, **State-Space Linear Memory ($O(1)$)**, **Bare-Metal Zero-Copy Compilation**, and **Dense Semantic Bytecode** presents a viable, physically grounded pathway to break free from the compute and thermal walls of mobile hardware. NanoLang transforms edge AI from a downscaled server artifact into a native mobile technology.
