"""
train.py – Training loop for the policy-value network.

Loss = cross_entropy(policy_logits, policy_target)
     + value_loss_weight × MSE(value, value_target)

Cross-entropy uses soft MCTS visit-count targets (not one-hot).

OPTIMISATION: D4 DATA AUGMENTATION
------------------------------------
The network architecture symmetrises outputs by averaging over all 8
D4 group elements at inference time.  During training we reinforce this
by applying a *random* D4 transform to every mini-batch before computing
gradients.

Why this helps
~~~~~~~~~~~~~~
  1.  Effective dataset size: each example is seen in 8 orientations
      across training steps, giving the network 8× more diverse signal
      per stored position without any extra self-play cost.
  2.  Regularisation: the augmentation acts as a strong prior that the
      value and policy should be orientation-invariant, matching the
      inductive bias already built into the network.
  3.  Faster convergence: with fewer training iterations needed per
      dataset pass, we can reduce `train_steps_per_iter` or get better
      policy quality in the same number of steps.

Implementation
~~~~~~~~~~~~~~
  `_apply_d4_np(arr, g)` applies element g of D4 to the spatial
  dimensions of a NumPy array.  It mirrors `apply_d4` in model.py but
  works on NumPy arrays using `np.flip` and `np.rot90`.

  `augment_batch` selects a random g ∈ {0..7} and transforms both the
  state tensor and the policy target consistently.  The value target
  is orientation-invariant so it is left unchanged.

  Correctness: after a rotation/flip by g the policy cell that was at
  flat index i in the original board is now at a different flat index
  i' = transform(i).  We achieve the correct re-indexing by reshaping
  the policy to (B, H, W), applying the same spatial transform, then
  flattening again.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import CFG
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# NumPy D4 helpers  (mirror of model.py's torch version)
# ---------------------------------------------------------------------------

def _apply_d4_np(arr: np.ndarray, g: int) -> np.ndarray:
    """
    Apply D4 group element g to a NumPy array with spatial dims at (-2, -1).

    Convention (matching model.py):
      g ∈ [0, 3]  →  counter-clockwise rotation by g×90 degrees
      g ∈ [4, 7]  →  horizontal flip, then rotation by (g−4)×90 degrees
    """
    if g >= 4:
        arr = np.flip(arr, axis=-1)   # horizontal mirror (left-right)
        g  -= 4
    if g > 0:
        # np.rot90 with axes=(-2, -1) rotates the H×W plane
        arr = np.rot90(arr, k=g, axes=(-2, -1))
    return arr


def augment_batch(
    states:   np.ndarray,   # (B, C, H, W)  float32
    policies: np.ndarray,   # (B, H*W)      float32
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply one randomly chosen D4 transform to the entire mini-batch.

    The *same* transform g is applied to both states and policies so
    that they remain consistent.  Values are orientation-invariant
    and are returned unchanged by the caller.

    Returns
    -------
    states_t, policies_t  – transformed copies (contiguous float32)
    """
    g = random.randint(0, 7)

    if g == 0:
        # Identity: return originals unchanged (no copy needed)
        return states, policies

    # Transform state channels – spatial dims are the last two axes
    states_t = np.ascontiguousarray(_apply_d4_np(states, g))

    # Reshape policy to (B, H, W), transform, flatten back
    policies_2d = policies.reshape(-1, rows, cols)
    policies_t  = np.ascontiguousarray(
        _apply_d4_np(policies_2d, g).reshape(-1, rows * cols)
    )

    return states_t, policies_t


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model:             torch.nn.Module,
    optimizer:         torch.optim.Optimizer,
    states:            torch.Tensor,   # (B, C, H, W)
    policy_targets:    torch.Tensor,   # (B, n_actions)
    value_targets:     torch.Tensor,   # (B,)
    value_loss_weight: float = CFG.value_loss_weight,
) -> dict[str, float]:
    """One gradient update.  Returns dict with 'loss', 'policy_loss', 'value_loss'."""
    model.train()
    policy_logits, value_pred = model(states)

    # Soft cross-entropy: −Σ p_target · log p_pred
    log_probs   = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()

    value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets)
    loss       = policy_loss + value_loss_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss":        loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss":  value_loss.item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model:       torch.nn.Module,
    optimizer:   torch.optim.Optimizer,
    buffer:      ReplayBuffer,
    device:      torch.device,
    num_steps:   int   = CFG.train_steps_per_iter,
    batch_size:  int   = CFG.batch_size,
    log_interval: int  = CFG.log_interval,
) -> dict[str, float]:
    """
    Run *num_steps* gradient updates sampling from *buffer*.

    Each mini-batch is augmented with a random D4 transform before the
    forward pass (see module docstring).

    Returns mean losses over all steps.
    """
    if len(buffer) < batch_size:
        print(f"  [train] Buffer too small ({len(buffer)} < {batch_size}), skipping.")
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    total = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    for step in range(1, num_steps + 1):
        states_np, policies_np, values_np = buffer.sample(batch_size)

        # ── D4 augmentation ───────────────────────────────────────────
        # Apply a single random group element to the whole batch.
        # The transform is applied in NumPy before converting to Torch,
        # so it adds virtually no GPU/CPU compute cost.
        states_np, policies_np = augment_batch(
            states_np, policies_np, CFG.rows, CFG.cols
        )
        # ──────────────────────────────────────────────────────────────

        states   = torch.from_numpy(states_np).to(device)
        policies = torch.from_numpy(policies_np).to(device)
        values   = torch.from_numpy(values_np).to(device)

        metrics = train_step(model, optimizer, states, policies, values)
        for k, v in metrics.items():
            total[k] += v

        if step % log_interval == 0 or step == num_steps:
            avg = {k: v / step for k, v in total.items()}
            print(
                f"  [train] step {step:4d}/{num_steps} | "
                f"loss={avg['loss']:.4f}  "
                f"policy={avg['policy_loss']:.4f}  "
                f"value={avg['value_loss']:.4f}"
            )

    return {k: v / num_steps for k, v in total.items()}


# ---------------------------------------------------------------------------
# Optimizer factory   (unchanged)
# ---------------------------------------------------------------------------

def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Adam with weight decay."""
    return Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
