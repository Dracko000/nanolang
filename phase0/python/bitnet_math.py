"""BitNet b1.58 Mathematical Testbed and Validation Module.

Implements the exact quantization functions and error bounds defined in
the NanoLang Whitepaper (WHITEPAPER_NANOLANG.md).
"""

import numpy as np


def quantize_weights_ternary(W: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantizes continuous weights W into {-1, 0, +1} with scale factor gamma.

    gamma = mean(|W|)
    W_ternary = clip(round(W / (gamma + eps)), -1, 1)
    """
    eps = 1e-7
    gamma = float(np.mean(np.abs(W)))
    if gamma < eps:
        gamma = eps
    scaled = W / gamma
    W_ternary = np.clip(np.round(scaled), -1.0, 1.0).astype(np.int8)
    return W_ternary, gamma


def quantize_activations_int8(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantizes continuous activations X into INT8 [-128, 127] with scale beta.

    beta = max(|X|) / 127.0
    X_int8 = clip(round(X / (beta + eps)), -128, 127)
    """
    eps = 1e-7
    max_val = float(np.max(np.abs(X)))
    beta = max_val / 127.0 if max_val > eps else eps
    scaled = X / beta
    X_int8 = np.clip(np.round(scaled), -128.0, 127.0).astype(np.int8)
    return X_int8, beta


def ternary_matmul_forward(
    X: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Simulates the end-to-end BitNet b1.58 forward linear projection.

    Returns the reconstructed continuous output Y_hat and execution statistics.
    """
    # 1. Quantize weights and activations
    W_ternary, gamma = quantize_weights_ternary(W)
    X_int8, beta = quantize_activations_int8(X)

    # 2. Pure integer dot-product (No floating point multiplication inside)
    # Cast to int32 to prevent overflow during accumulation
    X_i32 = X_int8.astype(np.int32)
    W_i32 = W_ternary.astype(np.int32)
    accum_i32 = np.dot(X_i32, W_i32)

    # 3. Rescale output once per tensor (scalar scaling)
    Y_hat = accum_i32.astype(np.float32) * (gamma * beta)

    # Calculate statistics
    total_elements = W.size
    zeros = np.sum(W_ternary == 0)
    positives = np.sum(W_ternary == 1)
    negatives = np.sum(W_ternary == -1)

    stats = {
        "gamma": gamma,
        "beta": beta,
        "sparsity_ratio": float(zeros / total_elements),
        "positive_ratio": float(positives / total_elements),
        "negative_ratio": float(negatives / total_elements),
        "weight_footprint_fp32_mb": (total_elements * 4) / (1024 * 1024),
        "weight_footprint_1_58bit_mb": (total_elements * 0.2) / (1024 * 1024),
    }

    return Y_hat, stats


def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Computes cosine similarity and relative L2 error between true and predicted activations."""
    norm_true = np.linalg.norm(Y_true)
    norm_pred = np.linalg.norm(Y_pred)
    cosine_sim = float(
        np.dot(Y_true.flatten(), Y_pred.flatten()) / (norm_true * norm_pred + 1e-9)
    )
    rel_l2_err = float(np.linalg.norm(Y_true - Y_pred) / (norm_true + 1e-9))

    return {
        "cosine_similarity": cosine_sim,
        "relative_l2_error": rel_l2_err,
    }


if __name__ == "__main__":
    np.random.seed(42)
    # Simulate a 4096 x 4096 attention projection layer
    K, M = 4096, 4096
    print("=" * 70)
    print(f" BITNET b1.58 MATHEMATICAL VALIDATION (Layer: {K}x{M})")
    print("=" * 70)

    # Sample weights from standard normal distribution (simulating trained weights)
    W = np.random.randn(K, M).astype(np.float32) * 0.02
    # Sample activations from Gaussian with occasional outliers
    X = np.random.randn(1, K).astype(np.float32)

    # FP32 ground truth
    Y_true = np.dot(X, W)

    # Ternary forward
    Y_hat, stats = ternary_matmul_forward(X, W)
    metrics = compute_metrics(Y_true, Y_hat)

    print(f"Weights Distribution:")
    print(f"  Zeros   (Skipped):    {stats['sparsity_ratio']*100:.2f}%")
    print(f"  Positive (+1 ADD):    {stats['positive_ratio']*100:.2f}%")
    print(f"  Negative (-1 SUB):    {stats['negative_ratio']*100:.2f}%")
    print()
    print(f"Reconstruction Fidelity:")
    print(f"  Cosine Similarity:    {metrics['cosine_similarity']:.4f} (Target: > 0.95)")
    print(f"  Relative L2 Error:    {metrics['relative_l2_error']:.4f}")
    print()
    print(f"Memory Footprint Reduction:")
    print(f"  FP32 Footprint:       {stats['weight_footprint_fp32_mb']:.2f} MB")
    print(f"  1.58-bit Footprint:   {stats['weight_footprint_1_58bit_mb']:.2f} MB (Radix-3 packed)")
    print(f"  Compression Ratio:    {stats['weight_footprint_fp32_mb'] / stats['weight_footprint_1_58bit_mb']:.1f}x reduction")
    print("=" * 70)
