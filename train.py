"""
train.py – Training loop for the policy-value network.

Loss = cross_entropy(policy_logits, policy_target)
     + value_loss_weight * MSE(value, value_target)

Cross-entropy is computed manually (log-softmax + dot with target
distribution) so we support soft targets from MCTS visit counts,
not just one-hot labels.

The illegal-action mask is NOT re-applied during training: the network
learns to assign low probability to illegal cells from the data alone,
since MCTS targets are always zero for illegal actions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import CFG
from replay_buffer import ReplayBuffer


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    value_loss_weight: float = CFG.value_loss_weight,
) -> dict[str, float]:
    """
    Perform one gradient update.

    Parameters
    ----------
    model           : The network.
    optimizer       : Torch optimiser.
    states          : (B, C, H, W) float32.
    policy_targets  : (B, n_actions) float32 – MCTS visit-count distribution.
    value_targets   : (B,) float32 – +1 / -1 game outcome.
    value_loss_weight: Weight of the value loss term.

    Returns
    -------
    dict with keys 'loss', 'policy_loss', 'value_loss'.
    """
    model.train()
    policy_logits, value_pred = model(states)

    # Policy loss: cross-entropy with soft target distribution.
    # log_softmax of logits · policy_target (negative sum = CE).
    log_probs = F.log_softmax(policy_logits, dim=-1)      # (B, n_actions)
    policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()

    # Value loss: MSE between predicted scalar and {+1, -1} target.
    value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets)

    loss = policy_loss + value_loss_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    }


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    num_steps: int = CFG.train_steps_per_iter,
    batch_size: int = CFG.batch_size,
    log_interval: int = CFG.log_interval,
) -> dict[str, float]:
    """
    Run *num_steps* gradient updates sampling from *buffer*.

    Returns
    -------
    dict with mean losses over all steps.
    """
    if len(buffer) < batch_size:
        print(f"  [train] Buffer too small ({len(buffer)} < {batch_size}), skipping.")
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    total = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    for step in range(1, num_steps + 1):
        states_np, policies_np, values_np = buffer.sample(batch_size)

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


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create the Adam optimiser with weight decay."""
    return Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
