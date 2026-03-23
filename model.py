"""
model.py – Small CNN with a shared trunk and two heads:

  policy_logits : (batch, rows * cols)  – raw logits over all cells
  value         : (batch, 1)            – game outcome prediction in [-1, 1]

Architecture: input → [Conv BN ReLU] × 1 trunk → [ResBlock] × K → heads

Design choices:
- Residual blocks stabilise training without being overly complex.
- BatchNorm after each conv for stable gradients.
- Policy head: 1×1 conv → flatten → linear.
- Value head: 1×1 conv → flatten → linear → tanh.
- No dropout – the stochastic MCTS targets already act as regularisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


class ResBlock(nn.Module):
    """A simple residual block: two 3×3 convolutions with a skip connection."""

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ChainReactionNet(nn.Module):
    """
    Policy-Value network for Chain Reaction.

    Parameters
    ----------
    rows, cols       : Board dimensions.
    in_channels      : Number of input feature planes (default from CFG).
    num_filters      : Feature maps in each conv layer.
    num_res_blocks   : Number of residual blocks in the trunk.
    """

    def __init__(
        self,
        rows: int = CFG.rows,
        cols: int = CFG.cols,
        in_channels: int = CFG.num_channels,
        num_filters: int = CFG.num_filters,
        num_res_blocks: int = CFG.num_res_blocks,
    ) -> None:
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.n_actions = rows * cols

        # ── Input stem ───────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # ── Residual trunk ───────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # ── Policy head ──────────────────────────────────────────────
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * rows * cols, self.n_actions)

        # ── Value head ───────────────────────────────────────────────
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, channels, rows, cols) float32 tensor

        Returns
        -------
        policy_logits : (batch, rows * cols)
        value         : (batch, 1)  in [-1, 1]
        """
        h = self.stem(x)
        h = self.res_blocks(h)

        # Policy
        p = self.policy_conv(h)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = self.value_conv(h)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


# ---------------------------------------------------------------------------
# Helper: build a model and move it to the appropriate device
# ---------------------------------------------------------------------------

def build_model(device: torch.device | None = None) -> tuple[ChainReactionNet, torch.device]:
    """
    Instantiate the network, move it to *device* (auto-detect if None),
    and return (model, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChainReactionNet().to(device)
    return model, device
