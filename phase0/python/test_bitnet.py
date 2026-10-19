import unittest
import numpy as np
from bitnet_math import (
    quantize_weights_ternary,
    quantize_activations_int8,
    ternary_matmul_forward,
    compute_metrics,
)


class TestBitNetMath(unittest.TestCase):
    def setUp(self):
        np.random.seed(1337)

    def test_weight_quantization_bounds(self):
        """Weights must only be -1, 0, or 1."""
        W = np.random.randn(256, 256).astype(np.float32)
        W_ternary, gamma = quantize_weights_ternary(W)
        unique_vals = set(np.unique(W_ternary))
        self.assertTrue(unique_vals.issubset({-1, 0, 1}))
        self.assertGreater(gamma, 0.0)

    def test_activation_quantization_bounds(self):
        """Activations must be within INT8 range [-128, 127]."""
        X = np.random.randn(32, 256).astype(np.float32) * 5.0
        X_int8, beta = quantize_activations_int8(X)
        self.assertGreaterEqual(np.min(X_int8), -128)
        self.assertLessEqual(np.max(X_int8), 127)
        self.assertGreater(beta, 0.0)

    def test_reconstruction_correlation(self):
        """Ternary forward output must strongly correlate with FP32 ground truth."""
        K, M = 1024, 1024
        W = np.random.randn(K, M).astype(np.float32) * 0.02
        X = np.random.randn(4, K).astype(np.float32)

        Y_true = np.dot(X, W)
        Y_hat, stats = ternary_matmul_forward(X, W)
        metrics = compute_metrics(Y_true, Y_hat)

        # On untrained random Gaussian weights, a 3-level ternary quantizer
        # mathematically achieves ~0.88 cosine similarity (trained BitNet weights
        # with straight-through estimator reach > 0.98).
        self.assertGreater(metrics["cosine_similarity"], 0.85)

    def test_sparsity_distribution(self):
        """Gaussian weights centered at 0 should have healthy ternary distribution."""
        W = np.random.randn(512, 512).astype(np.float32)
        W_ternary, _ = quantize_weights_ternary(W)
        zeros = np.mean(W_ternary == 0)
        pos = np.mean(W_ternary == 1)
        neg = np.mean(W_ternary == -1)

        # Standard normal distribution clipped with mean scale has ~38% zeros and ~31% pos/neg
        self.assertGreater(zeros, 0.20)
        self.assertGreater(pos, 0.20)
        self.assertGreater(neg, 0.20)


if __name__ == "__main__":
    unittest.main()
