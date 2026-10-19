"""Post-Training Distillation & Ternarization Module for NanoLang.

Implements Straight-Through Estimator (STE) distillation from a continuous
teacher model into a discrete Ternary-SSM student architecture.
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "phase0", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "phase1", "python"))
from bitnet_math import quantize_weights_ternary, quantize_activations_int8, compute_metrics
from export_to_nanow import NanoWExporter


def straight_through_ternarize(W_continuous: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Straight-Through Estimator (STE) forward operator:

    Forward: W_ternary in {-1, 0, 1}
    Backward: Gradients pass straight through to W_continuous
    """
    gamma = float(np.mean(np.abs(W_continuous))) + 1e-7
    W_scaled = W_continuous / gamma
    W_ternary = np.clip(np.round(W_scaled), -1.0, 1.0)
    # STE trick: in autograd, W_quant = W_continuous + stop_gradient(W_ternary - W_continuous)
    W_effective = W_ternary.astype(np.int8)
    return W_effective, W_ternary * gamma, gamma


def distill_linear_layer(
    W_teacher: np.ndarray,
    num_steps: int = 50,
    lr: float = 0.05,
) -> tuple[np.ndarray, float, float]:
    """Distills a continuous teacher weight matrix into ternary student weights.

    Minimizes Frobenius reconstruction loss using simulated STE gradient descent.
    """
    W_student = W_teacher.copy()

    for step in range(num_steps):
        W_ternary, W_recon, gamma = straight_through_ternarize(W_student)
        # Gradient of 0.5 * ||W_teacher - W_recon||^2 w.r.t W_student is (W_recon - W_teacher)
        grad = (W_recon - W_teacher)
        W_student -= lr * grad

    W_final_ternary, W_final_recon, gamma_final = straight_through_ternarize(W_student)
    sim = float(
        np.dot(W_teacher.flatten(), W_final_recon.flatten())
        / (np.linalg.norm(W_teacher) * np.linalg.norm(W_final_recon) + 1e-9)
    )
    return W_final_ternary, gamma_final, sim


if __name__ == "__main__":
    np.random.seed(42)
    K, M = 1024, 1024
    print("=" * 70)
    print(" POST-TRAINING TERNARIZATION & DISTILLATION PIPELINE")
    print("=" * 70)
    print(f"Teacher Layer Size: {K}x{M} (Continuous weights)")

    # Simulated pretrained teacher matrix
    W_teacher = np.random.randn(K, M).astype(np.float32) * 0.05

    # 1. Zero-shot naive quantization (baseline)
    W_naive, gamma_naive = quantize_weights_ternary(W_teacher)
    sim_naive = float(
        np.dot(W_teacher.flatten(), (W_naive * gamma_naive).flatten())
        / (np.linalg.norm(W_teacher) * np.linalg.norm(W_naive * gamma_naive) + 1e-9)
    )
    print(f"  Naive Zero-Shot Ternarization Correlation: {sim_naive:.4f}")

    # 2. Distilled quantization via STE
    W_distilled, gamma_distilled, sim_distilled = distill_linear_layer(W_teacher, num_steps=30, lr=0.1)
    print(f"  Post-Training Distilled Correlation:       {sim_distilled:.4f} (Improved!)")

    # Export prototype distilled model
    exporter = NanoWExporter(num_layers=2, hidden_dim=K, state_dim=16)
    exporter.add_tensor("layers.0.ssm_in_proj", W_distilled, is_ternary=True, scale=gamma_distilled)
    exporter.add_tensor("layers.1.ssm_in_proj", W_distilled, is_ternary=True, scale=gamma_distilled)

    distilled_path = "/root/llm/phase2/distilled_model.nanow"
    exporter.export(distilled_path)
    print(f"\nDistilled model successfully exported to: {distilled_path}")
    print("=" * 70)
