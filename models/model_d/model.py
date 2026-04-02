"""
model.py  –  Random baseline model for model_d.

forward() returns fresh torch.randn logits on every call, giving MCTS
no consistent signal → effectively random move selection.

The model has a single dummy parameter so state_dict save/load works
identically to any other model.  No changes to tournament.py needed.
"""

import torch
import torch.nn as nn
from config import CFG


class ChainReactionNet(nn.Module):

    def __init__(
        self,
        rows:        int = CFG.rows,
        cols:        int = CFG.cols,
        **kwargs,               # absorbs num_filters etc. if called generically
    ) -> None:
        super().__init__()
        self.rows      = rows
        self.cols      = cols
        self.n_actions = rows * cols

        # One dummy parameter so state_dict is non-empty and
        # torch.save / load_state_dict round-trips cleanly.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        N = self.n_actions
        # New random logits every call → MCTS gets no consistent guidance
        # → play is effectively random regardless of num_simulations.
        policy_logits = torch.randn(B, N, device=x.device)
        value         = torch.zeros(B, 1,  device=x.device)
        return policy_logits, value


def build_model(
    device:        torch.device | None = None,
    compile_model: bool = False,          # accepted for API compatibility
) -> tuple[ChainReactionNet, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChainReactionNet().to(device)
    return model, device