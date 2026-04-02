"""
model.py  –  CR-EGNN: Chain Reaction Equivariant Graph Neural Network
======================================================================

Why a GNN, and why is it D4-equivariant?
-----------------------------------------
The 5×5 board is a graph: 25 nodes (cells), edges between neighbouring cells.
A message-passing neural network (MPNN) with

  • shared weights across all nodes
  • a symmetric (permutation-invariant) aggregation function (sum)

is equivariant to every automorphism of the input graph.  The automorphism
group of the 5×5 grid graph is exactly D4 (four rotations × two reflections).
Equivariance therefore holds at every intermediate layer — not just at the
output — with a single forward pass and no 8-fold averaging overhead.

Structural alignment with game physics
---------------------------------------
A Chain Reaction explosion distributes orbs from one cell to all its
neighbours.  This is literally one round of neighbourhood aggregation.
K message-passing layers give the network an explicit representational
budget for chain reactions of depth ≤ K.

Differences from the prior ResNet model
-----------------------------------------
  Old: 8 forward passes → average outputs → approximate D4 output-invariance.
  New: 1 forward pass   → true D4 equivariance at every layer.

  Old: BatchNorm   → mode-switch required; statistics meaningless at batch=1.
  New: LayerNorm   → identical behaviour at any batch size (MCTS + training).

  Old: ~700 K parameters.
  New: ~180 K parameters (h=128, K=6).

  Old: ~4–8× slower per MCTS call (8 forward passes).
  New: single pass; 25×25 adjacency matrix lives permanently in L1 cache.

API compatibility
------------------
  build_model()  – same signature as before.
  ChainReactionNet.forward(x)  – same input/output contract:
      x              : (B, C, H, W)  produced by encode_state_tensor()
      policy_logits  : (B, N)        one logit per board cell
      value          : (B, 1)        scalar in (−1, 1)

No other file needs to change.
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn

from config import CFG


# ---------------------------------------------------------------------------
# Static graph construction
# ---------------------------------------------------------------------------

def build_adjacency(rows: int, cols: int) -> torch.Tensor:
    """
    Build the symmetric 0/1 adjacency matrix for the board graph.

    The matrix is computed once at model construction and registered as a
    non-parameter buffer so it:
      • moves to the correct device automatically with model.to(device)
      • is included in state_dict for seamless checkpoint save/load
      • is never recomputed during training or inference

    Returns
    -------
    torch.Tensor, shape (N, N), dtype float32, where N = rows * cols.
    A[i, j] = 1.0  iff cells i and j share a border (4-connectivity).
    """
    n = rows * cols
    A = torch.zeros(n, n, dtype=torch.float32)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    A[i, nr * cols + nc] = 1.0
    return A


# ---------------------------------------------------------------------------
# Message-passing layer
# ---------------------------------------------------------------------------

class MPLayer(nn.Module):
    """
    One round of equivariant message passing.

    Pipeline
    --------
    1. Message   : apply a shared 2-layer MLP to every node's current features.
    2. Aggregate : sum incoming messages over all neighbours via adj @ messages.
    3. Update    : GRUCell(input=aggregated, hidden=current) — gated update
                   prevents gradient vanishing through K=6 layers.
    4. Normalise : LayerNorm over the feature dimension.

    Equivariance guarantee
    ----------------------
    All three operations (MLP, sum aggregation via A, GRU) share their
    parameters across every node.  Together with a symmetric aggregation
    function, this makes the layer equivariant to any permutation π such
    that A = π A πᵀ — i.e., every graph automorphism, which for our board
    graph is exactly D4.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

        # Message function: 2-layer MLP, shared across all nodes.
        # Produces one message vector per node from that node's features.
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        # Gated node update: GRUCell(input=aggregated_neighbours,
        #                            hidden=own_current_state)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Post-update normalisation: LayerNorm over feature dim.
        # Works identically at batch size 1 (MCTS) and large batches (training).
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h   : (B, N, D)  current node feature matrix.
        adj : (N, N)     precomputed adjacency buffer (float32, 0/1).

        Returns
        -------
        h_new : (B, N, D)  updated node features.
        """
        B, N, D = h.shape

        # ── 1. Message ────────────────────────────────────────────────
        # Apply shared MLP to every node's features simultaneously.
        m = self.message_mlp(h)                          # (B, N, D)

        # ── 2. Aggregate ──────────────────────────────────────────────
        # sum_{j ∈ neighbours(i)} m_j
        # einsum: "nm, bmd -> bnd"
        #   n = target node index
        #   m = source node index  (adj[n, m] = 1 if m is neighbour of n)
        #   b = batch index,  d = feature index
        agg = torch.einsum("nm, bmd -> bnd", adj, m)    # (B, N, D)

        # ── 3. GRU node update ────────────────────────────────────────
        # GRUCell operates on (batch, input) pairs; flatten B and N,
        # then restore shape.
        h_flat   = h.reshape(B * N, D)
        agg_flat = agg.reshape(B * N, D)
        h_new    = self.gru(agg_flat, h_flat).reshape(B, N, D)

        # ── 4. LayerNorm ──────────────────────────────────────────────
        return self.norm(h_new)                          # (B, N, D)


# ---------------------------------------------------------------------------
# Full CR-EGNN
# ---------------------------------------------------------------------------

class ChainReactionNet(nn.Module):
    """
    CR-EGNN  –  Chain Reaction Equivariant Graph Neural Network.

    Parameters
    ----------
    rows, cols    : Board dimensions (must be equal for D4 symmetry).
    in_channels   : Number of input feature channels per cell (default 5).
    hidden_dim    : Width of node feature vectors in every MP layer (default 128).
    num_layers    : Number of message-passing rounds (default 6).
    """

    def __init__(
        self,
        rows:        int = CFG.rows,
        cols:        int = CFG.cols,
        in_channels: int = CFG.num_channels,
        hidden_dim:  int = CFG.gnn_hidden_dim,
        num_layers:  int = CFG.gnn_num_layers,
    ) -> None:
        super().__init__()

        self.rows      = rows
        self.cols      = cols
        self.n         = rows * cols
        self.n_actions = self.n   # kept for compatibility with MCTS / eval code

        # Pre-compute adjacency and register as a non-parameter buffer.
        # It will be automatically moved to the right device by .to(device)
        # and included in state_dict for checkpoint save/load.
        self.register_buffer("adj", build_adjacency(rows, cols))  # (N, N)

        # ── Node embedding ────────────────────────────────────────────
        # Project raw per-cell features to the hidden space.
        # LayerNorm before ReLU stabilises early training.
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Message-passing layers ────────────────────────────────────
        # Independent parameter sets per layer (no weight tying) so each
        # layer can learn a qualitatively different transformation.
        self.mp_layers = nn.ModuleList(
            [MPLayer(hidden_dim) for _ in range(num_layers)]
        )

        # ── Policy head ───────────────────────────────────────────────
        # Shared linear projection: hidden_dim → 1 scalar per node.
        # Because weights are shared and inputs are D4-equivariant,
        # the output logits are also D4-equivariant.
        self.policy_head = nn.Linear(hidden_dim, 1, bias=True)

        # ── Value head ────────────────────────────────────────────────
        # Sum pooling (D4-invariant) followed by a 2-layer MLP.
        # Sum (not mean) preserves total material count as a signal.
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, C, H, W)  – encoded board state from encode_state_tensor().

        Returns
        -------
        policy_logits : (B, N)  – one logit per cell, D4-equivariant.
        value         : (B, 1)  – scalar in (−1, 1), D4-invariant.
        """
        B = x.size(0)

        # ── (B, C, H, W) → (B, N, C) ─────────────────────────────────
        # Treat every cell as a graph node with C input features.
        # permute: (B, C, H, W) → (B, H, W, C)
        # reshape:              → (B, N, C)
        h = x.permute(0, 2, 3, 1).reshape(B, self.n, -1)   # (B, N, C)

        # ── Initial embedding ─────────────────────────────────────────
        h = self.embedding(h)                               # (B, N, hidden_dim)

        # ── Message passing ───────────────────────────────────────────
        # self.adj is automatically on the correct device (registered buffer).
        for layer in self.mp_layers:
            h = layer(h, self.adj)                          # (B, N, hidden_dim)

        # ── Policy head ───────────────────────────────────────────────
        # policy_head is a Linear(hidden_dim, 1) shared across all nodes.
        # squeeze(-1) removes the trailing singleton dimension.
        policy_logits = self.policy_head(h).squeeze(-1)     # (B, N)

        # ── Value head ────────────────────────────────────────────────
        # Sum over all N nodes → D4-invariant global representation.
        v     = h.sum(dim=1)                                # (B, hidden_dim)
        value = torch.tanh(self.value_mlp(v))               # (B, 1)

        return policy_logits, value


# ---------------------------------------------------------------------------
# Factory  (same public signature as the old model.py)
# ---------------------------------------------------------------------------

def build_model(
    device:        torch.device | None = None,
    compile_model: bool                = False,
) -> tuple[ChainReactionNet, torch.device]:
    """
    Build and return (model, device).

    Drop-in replacement for the original build_model().  All callers
    (main.py, selfplay.py, evaluate.py) use this function unchanged.

    compile_model
    -------------
    When True and supported, wraps the model with torch.compile().
    Disabled on Windows where TorchInductor may be unavailable.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChainReactionNet(
        rows=CFG.rows,
        cols=CFG.cols,
        in_channels=CFG.num_channels,
        hidden_dim=CFG.gnn_hidden_dim,
        num_layers=CFG.gnn_num_layers,
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
