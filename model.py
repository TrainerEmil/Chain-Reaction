"""
model.py – AlphaZero policy-value network for Chain Reaction.

D4-symmetry averaging
---------------------
The original implementation called `self.base(x_g)` **eight times** in
a Python for-loop.  On CPU each call carries a large fixed overhead
(Python dispatch, BatchNorm statistics, memory allocation).
All 8 transformed inputs are concatenated into a single
tensor of shape (8·B, C, H, W) and passed through `base` in **one**
forward call.  The result is split and the inverse transforms are
applied before averaging.

torch.compile
-------------
`build_model` compiles the model with `torch.compile(mode='reduce-overhead')`
when running on CPU and PyTorch ≥ 2.0.  The first call incurs a one-time
compilation cost (~a few seconds); every subsequent call is faster.
Use `compile_model=False` to disable (e.g. for unit tests).
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG

# ---------------------------------------------------------------------------
# D4 spatial transforms
# ---------------------------------------------------------------------------

def rot90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k=k, dims=(-2, -1))


def flip_h(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-1,))


def apply_d4(x: torch.Tensor, g: int) -> torch.Tensor:
    """
    g in [0..7]
      0..3 : rotations by 0, 90, 180, 270 degrees
      4..7 : horizontal flip followed by rotations 0, 90, 180, 270
    Works on tensors of shape (..., H, W).
    """
    if g < 4:
        return rot90(x, g)
    return rot90(flip_h(x), g - 4)


def invert_d4(g: int) -> int:
    """
    Inverse element for the D4 group with our parameterisation.
      inverse(rotation k)         = rotation (−k) mod 4
      inverse(flip then rotation k) = flip then rotation k   (self-inverse)
    """
    if g < 4:
        return (-g) % 4
    return g


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class BaseEquiBackbone(nn.Module):
    def __init__(
        self,
        in_channels:    int = CFG.num_channels,
        num_filters:    int = 96,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head: one logit per board cell
        self.policy_head = nn.Conv2d(num_filters, 1, kernel_size=1, bias=True)

        # Value head: invariant global average pooling
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_filters // 2),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(num_filters // 2, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x          : (B, C, H, W)
        policy_map : (B, H, W)   – one logit per cell
        value      : (B, 1)      – scalar in (−1, 1)
        """
        h = self.stem(x)
        h = self.trunk(h)

        policy_map = self.policy_head(h).squeeze(1)   # (B, H, W)

        v = self.value_conv(h).mean(dim=(-2, -1))     # global average pool → (B, C//2)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))         # (B, 1)

        return policy_map, value


# ---------------------------------------------------------------------------
# D4-symmetrized policy-value network
# ---------------------------------------------------------------------------

class ChainReactionNet(nn.Module):
    def __init__(
        self,
        rows:           int = CFG.rows,
        cols:           int = CFG.cols,
        in_channels:    int = CFG.num_channels,
        num_filters:    int = 96,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()

        if rows != cols:
            raise ValueError(
                "D4 invariance with 90° rotations requires a square board."
            )

        self.rows      = rows
        self.cols      = cols
        self.n_actions = rows * cols
        self.base      = BaseEquiBackbone(
            in_channels=in_channels,
            num_filters=num_filters,
            num_res_blocks=num_res_blocks,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x              : (B, C, H, W)
        policy_logits  : (B, H*W)
        value          : (B, 1)
        """
        B = x.size(0)

        # ── Step 1: stack all 8 transforms ─────────────────────────────
        # list of 8 tensors, each (B, C, H, W) → cat → (8·B, C, H, W)
        x_all = torch.cat([apply_d4(x, g) for g in range(8)], dim=0)

        # ── Step 2: single forward pass through backbone ─────────────
        p_all, v_all = self.base(x_all)     # (8B, H, W)  and  (8B, 1)

        # ── Step 3: invert transforms on policy maps ──────────────────
        # Reshape to (8, B, H, W) so we can index by group element
        p_all = p_all.reshape(8, B, self.rows, self.cols)

        policy_map = torch.stack(
            [apply_d4(p_all[g], invert_d4(g)) for g in range(8)],
            dim=0,
        ).mean(dim=0)                       # (B, H, W)

        # ── Step 4: average value over group ─────────────────────────
        value = v_all.reshape(8, B, 1).mean(dim=0)   # (B, 1)

        policy_logits = policy_map.reshape(B, -1)    # (B, H*W)
        return policy_logits, value


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    device: torch.device | None = None,
    compile_model: bool = False,
):
    """
    Build and return (model, device).

    compile_model
    -------------
    When True and supported by the current platform/runtime, the model is
    compiled with torch.compile(mode='reduce-overhead').

    On Windows this is disabled by default because TorchInductor often
    requires an external C++ compiler that may not be installed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChainReactionNet(
        rows=CFG.rows,
        cols=CFG.cols,
        in_channels=CFG.num_channels,
        num_filters=CFG.num_filters,
        num_res_blocks=CFG.num_res_blocks,
    ).to(device)

    can_try_compile = (
        compile_model
        and hasattr(torch, "compile")
        and sys.platform != "win32"
    )

    if can_try_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  [model] torch.compile enabled  (mode='reduce-overhead')")
        except Exception as exc:
            print(f"  [model] torch.compile disabled, falling back to eager: {exc}")

    return model, device
