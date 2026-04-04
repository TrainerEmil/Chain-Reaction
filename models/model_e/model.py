"""
model.py  –  CR-CWN: Chain Reaction Cell-Complex W-Network
===========================================================

Background
----------
The 5×5 Chain Reaction board is a graph: 25 nodes (cells), 40 edges
(shared borders).  Your current CR-EGNN already exploits this graph
structure.  CR-CWN *lifts* that graph to a 2-dimensional cell complex by
additionally attaching 16 two-dimensional "plaquette" cells – one for
each 2×2 square on the board.

    Dimension  │  Cells           │  Count (5×5)  │  Input channels
    ───────────┼──────────────────┼───────────────┼─────────────────────
    0-cells    │  board squares   │  25 nodes     │  C_n = 5 (encoding)
    1-cells    │  shared borders  │  40 edges     │  C_e = 10 (sum+|diff|)
    2-cells    │  2×2 plaquettes  │  16 faces     │  C_f = 5  (corner mean)
    ───────────┴──────────────────┴───────────────┴─────────────────────

Cell complex topology (registered as non-parameter buffers)
------------------------------------------------------------
B₁ ∈ ℝ^{E×N}   signed edge-node incidence  (E=40, N=25)
B₂ ∈ ℝ^{F×E}   signed face-edge incidence  (F=16)
C_fn ∈ ℝ^{F×N}  corner-averaging matrix (C_fn[f,n] = 0.25 iff n is
                 a corner of face f)

All three satisfy the fundamental identity  B₂ B₁ = 0  (boundary of a
boundary is zero), which is the algebraic cornerstone of Hodge theory.

Hodge Laplacians (registered for analysis; not used directly in forward)
------------------------------------------------------------------------
L₀ = B₁ᵀ B₁ ∈ ℝ^{N×N}              — graph Laplacian on nodes
L₁ = B₁ B₁ᵀ + B₂ᵀ B₂ ∈ ℝ^{E×E}    — Hodge-1 Laplacian on edges
L₂ = B₂ B₂ᵀ ∈ ℝ^{F×F}              — face Laplacian

CWN message passing (one layer = 4 simultaneous passes)
-------------------------------------------------------
  (1)  Node → Edge   via  |B₁|ᵀ   lower-adjacency aggregation
  (2)  Face → Edge   via  |B₂|    upper-adjacency aggregation
  (3)  Edge → Node   via  |B₁|    lower-coboundary aggregation
  (4)  Edge → Face   via  |B₂|ᵀ  upper-coboundary aggregation

Unsigned incidence matrices are used for aggregation (sum of neighbours).
Signed B₁ is used only once, in the forward pass, to compute the initial
anti-symmetric edge feature  |B₁ x_n|  (the "gradient" across each border).

Updates: GRUCell(input=aggregated_msgs, hidden=current_features) + LayerNorm
         — identical to the CR-EGNN baseline.

D4 equivariance
---------------
The 5×5 grid complex is D4-symmetric: every rotation/reflection permutes
nodes, edges, and plaquettes in a consistent, structure-preserving way.
All four message-passing MLPs share weights across cells of the same
dimension, and the aggregation operators (|B₁|, |B₂|, etc.) are
equivariant under the induced D4 permutations of the complex.  Therefore
the full CR-CWN is D4-equivariant in a single forward pass.

API compatibility
-----------------
  build_model()                    same signature as the EGNN model.py
  ChainReactionNet.forward(x)      same input/output contract:
    x              : (B, C, H, W)  from encode_state_tensor()
    policy_logits  : (B, N)        one logit per board cell
    value          : (B, 1)        scalar in (−1, 1)
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn

from config import CFG


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def build_boundary_operators(
    rows: int, cols: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct signed boundary matrices B₁ (E×N) and B₂ (F×E).

    Edge indexing
    -------------
    Horizontal edge  (r, c) → (r, c+1):  index = r*(cols−1) + c
      (rows × (cols-1) edges, indices 0 … rows*(cols-1) − 1)
    Vertical edge    (r, c) → (r+1, c):  index = rows*(cols−1) + r*cols + c
      ((rows-1) × cols edges, indices rows*(cols-1) … E − 1)

    Face indexing
    -------------
    Plaquette with top-left corner at (r, c):  index = r*(cols−1) + c
      ((rows-1) × (cols-1) faces, indices 0 … F − 1)

    Sign conventions
    ----------------
    B₁  –  for edge e = u → v:  B₁[e, u] = −1,  B₁[e, v] = +1.
    B₂  –  plaquette (r, c) traversed counterclockwise (y-axis down):
              top edge  (r,c)→(r,c+1)         orientation matches  → +1
              right edge (r,c+1)→(r+1,c+1)   orientation matches  → +1
              bottom edge traversed right→left (against orientation) → −1
              left edge  traversed bottom→top  (against orientation) → −1
    This guarantees  B₂ B₁ = 0.

    Returns
    -------
    B1 : (E, N) float32
    B2 : (F, E) float32
    """
    N   = rows * cols
    n_h = rows * (cols - 1)          # number of horizontal edges
    n_v = (rows - 1) * cols          # number of vertical edges
    E   = n_h + n_v
    F   = (rows - 1) * (cols - 1)

    B1 = torch.zeros(E, N, dtype=torch.float32)
    B2 = torch.zeros(F, E, dtype=torch.float32)

    # ── B₁: horizontal edges ─────────────────────────────────────────
    for r in range(rows):
        for c in range(cols - 1):
            e = r * (cols - 1) + c
            B1[e, r * cols + c]     = -1.0   # source node
            B1[e, r * cols + c + 1] = +1.0   # target node

    # ── B₁: vertical edges ───────────────────────────────────────────
    for r in range(rows - 1):
        for c in range(cols):
            e = n_h + r * cols + c
            B1[e,  r      * cols + c] = -1.0
            B1[e, (r + 1) * cols + c] = +1.0

    # ── B₂: plaquette boundaries ──────────────────────────────────────
    for r in range(rows - 1):
        for c in range(cols - 1):
            f = r * (cols - 1) + c

            top_e   = r       * (cols - 1) + c         # horizontal, row r
            bot_e   = (r + 1) * (cols - 1) + c         # horizontal, row r+1
            right_e = n_h + r * cols + (c + 1)         # vertical,   col c+1
            left_e  = n_h + r * cols + c               # vertical,   col c

            B2[f, top_e]   = +1.0
            B2[f, right_e] = +1.0
            B2[f, bot_e]   = -1.0
            B2[f, left_e]  = -1.0

    return B1, B2


def build_corner_matrix(rows: int, cols: int) -> torch.Tensor:
    """
    Construct the face corner-averaging matrix C_fn ∈ ℝ^{F×N}.

    C_fn[f, n] = 0.25  if node n is one of the four corners of face f,
                  0.0   otherwise.

    Multiplying C_fn by a column of node scalars gives the mean of the
    four corner values for every face simultaneously:

        x_face = C_fn @ x_node   ∈ ℝ^{F×C}

    Returns
    -------
    C_fn : (F, N) float32
    """
    F   = (rows - 1) * (cols - 1)
    N   = rows * cols
    C   = torch.zeros(F, N, dtype=torch.float32)
    for r in range(rows - 1):
        for c in range(cols - 1):
            f = r * (cols - 1) + c
            for dr in range(2):
                for dc in range(2):
                    C[f, (r + dr) * cols + (c + dc)] = 0.25
    return C


# ---------------------------------------------------------------------------
# CWN layer
# ---------------------------------------------------------------------------

class CWNLayer(nn.Module):
    """
    One round of equivariant cell-complex message passing.

    Four message flows per round (all run simultaneously):

      (1) Node → Edge   :  m_n = MLP_n(h_n)
                           agg_e  += |B₁|ᵀ  m_n      (lower aggregation)
      (2) Face → Edge   :  m_f = MLP_f(h_f)
                           agg_e  += |B₂|   m_f       (upper aggregation)
      (3) Edge → Node   :  m_e_n = MLP_e→n(h_e)
                           agg_n  = |B₁|    m_e_n     (lower aggregation)
      (4) Edge → Face   :  m_e_f = MLP_e→f(h_e)
                           agg_f  = |B₂|ᵀ  m_e_f      (upper aggregation)

    Update rule (same as CR-EGNN MPLayer):
      h_*_new  = LayerNorm( GRUCell(input=agg_*, hidden=h_*) )

    The GRU gate allows the network to decide how much of the new
    multi-scale information to incorporate at each round, and helps
    prevent gradient vanishing over K layers.

    Equivariance note
    -----------------
    MLP weights are shared across ALL cells of the same dimension.
    The absolute-value incidence matrices are equivariant under any
    automorphism of the complex (in particular D4 for the grid).
    Together these properties make the layer D4-equivariant.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        D = hidden_dim

        # ── Message MLPs (2-layer, ReLU; shared across all cells of dim) ─
        def _mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(D, D, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(D, D, bias=True),
            )

        self.mlp_n      = _mlp()   # nodes   → edges   (flow 1)
        self.mlp_f      = _mlp()   # faces   → edges   (flow 2)
        self.mlp_e_to_n = _mlp()   # edges   → nodes   (flow 3)
        self.mlp_e_to_f = _mlp()   # edges   → faces   (flow 4)

        # ── Gated node/edge/face updates ──────────────────────────────
        self.gru_n = nn.GRUCell(D, D)
        self.gru_e = nn.GRUCell(D, D)
        self.gru_f = nn.GRUCell(D, D)

        # ── Post-update normalisation ──────────────────────────────────
        self.ln_n = nn.LayerNorm(D)
        self.ln_e = nn.LayerNorm(D)
        self.ln_f = nn.LayerNorm(D)

    def forward(
        self,
        h_n:      torch.Tensor,   # (B, N, D)  node features
        h_e:      torch.Tensor,   # (B, E, D)  edge features
        h_f:      torch.Tensor,   # (B, F, D)  face features
        B1_abs:   torch.Tensor,   # (E, N)     |B₁|
        B1_abs_T: torch.Tensor,   # (N, E)     |B₁|ᵀ
        B2_abs:   torch.Tensor,   # (F, E)     |B₂|
        B2_abs_T: torch.Tensor,   # (E, F)     |B₂|ᵀ
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, N, D = h_n.shape
        _,  E, _ = h_e.shape
        _,  F, _ = h_f.shape

        # ── 1. Prepare outgoing messages ──────────────────────────────
        m_n      = self.mlp_n(h_n)       # (B, N, D)
        m_f      = self.mlp_f(h_f)       # (B, F, D)
        m_e_to_n = self.mlp_e_to_n(h_e)  # (B, E, D)
        m_e_to_f = self.mlp_e_to_f(h_e)  # (B, E, D)

        # ── 2. Aggregate via unsigned boundary operators ───────────────
        # Edges receive from nodes (flow 1) AND faces (flow 2) – summed:
        agg_e = (
            torch.einsum("en, bnd -> bed", B1_abs,   m_n)     # (B, E, D)
            + torch.einsum("ef, bfd -> bed", B2_abs_T, m_f)   # (B, E, D)
        )

        # Nodes receive from edges (flow 3):
        agg_n = torch.einsum("ne, bed -> bnd", B1_abs_T, m_e_to_n)  # (B, N, D)

        # Faces receive from edges (flow 4):
        agg_f = torch.einsum("fe, bed -> bfd", B2_abs,   m_e_to_f)  # (B, F, D)

        # ── 3. GRU updates ─────────────────────────────────────────────
        # Flatten (B, X, D) → (B*X, D) for GRUCell, then restore shape.
        h_n_new = self.gru_n(
            agg_n.reshape(B * N, D), h_n.reshape(B * N, D)
        ).reshape(B, N, D)

        h_e_new = self.gru_e(
            agg_e.reshape(B * E, D), h_e.reshape(B * E, D)
        ).reshape(B, E, D)

        h_f_new = self.gru_f(
            agg_f.reshape(B * F, D), h_f.reshape(B * F, D)
        ).reshape(B, F, D)

        # ── 4. LayerNorm ──────────────────────────────────────────────
        return self.ln_n(h_n_new), self.ln_e(h_e_new), self.ln_f(h_f_new)


# ---------------------------------------------------------------------------
# Full CR-CWN
# ---------------------------------------------------------------------------

class ChainReactionNet(nn.Module):
    """
    CR-CWN – Chain Reaction Cell-Complex W-Network.

    Lifts the board graph to a 2-dimensional cell complex (nodes, edges,
    plaquettes) and runs K rounds of hierarchical four-way message passing
    across all three cell dimensions simultaneously.

    Parameters
    ----------
    rows, cols    : Board dimensions.
    in_channels   : Input feature channels per node (default 5).
    hidden_dim    : Feature vector width in every CWN layer.
    num_layers    : Number of CWN rounds.
    """

    def __init__(
        self,
        rows:        int = CFG.rows,
        cols:        int = CFG.cols,
        in_channels: int = CFG.num_channels,
        hidden_dim:  int = CFG.cwn_hidden_dim,
        num_layers:  int = CFG.cwn_num_layers,
    ) -> None:
        super().__init__()

        self.rows      = rows
        self.cols      = cols
        self.n_nodes   = rows * cols
        self.n_edges   = rows * (cols - 1) + (rows - 1) * cols
        self.n_faces   = (rows - 1) * (cols - 1)
        self.n_actions = self.n_nodes   # for MCTS / eval compatibility

        # ── Topology buffers ──────────────────────────────────────────
        # All matrices are fixed at construction and move to the correct
        # device automatically via .to(device) (register_buffer).
        B1, B2 = build_boundary_operators(rows, cols)
        C_fn   = build_corner_matrix(rows, cols)

        # Signed matrices (used only for initial feature derivation)
        self.register_buffer("B1", B1)                           # (E, N)
        self.register_buffer("B2", B2)                           # (F, E)

        # Unsigned matrices (used in message passing)
        B1_abs = B1.abs()
        B2_abs = B2.abs()
        self.register_buffer("B1_abs",   B1_abs)                 # (E, N)
        self.register_buffer("B1_abs_T", B1_abs.t().contiguous()) # (N, E)
        self.register_buffer("B2_abs",   B2_abs)                 # (F, E)
        self.register_buffer("B2_abs_T", B2_abs.t().contiguous()) # (E, F)

        # Corner-averaging matrix
        self.register_buffer("C_fn", C_fn)                       # (F, N)

        # ── Hodge Laplacians (buffers for analysis, not used in forward) ─
        # L₀ = B₁ᵀ B₁  ∈ ℝ^{N×N}  graph Laplacian on nodes
        # L₁ = B₁ B₁ᵀ + B₂ᵀ B₂  ∈ ℝ^{E×E}  Hodge-1 Laplacian on edges
        # L₂ = B₂ B₂ᵀ  ∈ ℝ^{F×F}  face Laplacian
        self.register_buffer("L0", B1.t() @ B1)
        self.register_buffer("L1", B1 @ B1.t() + B2.t() @ B2)
        self.register_buffer("L2", B2 @ B2.t())

        # ── Embedding layers ──────────────────────────────────────────
        # Initial edge features: concat(mean of endpoints, |diff of endpoints|)
        #   dimension = 2 * in_channels
        # Initial face features: mean of 4 corner node features
        #   dimension = in_channels
        in_ch_e = 2 * in_channels
        in_ch_f = in_channels

        def _embed(in_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_ch, hidden_dim, bias=True),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.embed_n = _embed(in_channels)   # node embedding
        self.embed_e = _embed(in_ch_e)       # edge embedding
        self.embed_f = _embed(in_ch_f)       # face embedding

        # ── CWN layers ────────────────────────────────────────────────
        self.cwn_layers = nn.ModuleList(
            [CWNLayer(hidden_dim) for _ in range(num_layers)]
        )

        # ── Policy head ───────────────────────────────────────────────
        # Linear(D → 1) shared across all nodes → D4-equivariant logits.
        self.policy_head = nn.Linear(hidden_dim, 1, bias=True)

        # ── Value head ────────────────────────────────────────────────
        # Sum-pool over all nodes (D4-invariant) then a 2-layer MLP.
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=True),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, C, H, W)  encoded board from encode_state_tensor().
            Identical format to the CR-EGNN baseline.

        Returns
        -------
        policy_logits : (B, N)   one logit per board cell (D4-equivariant).
        value         : (B, 1)   scalar in (−1, 1)        (D4-invariant).
        """
        B = x.size(0)

        # ── (B, C, H, W) → (B, N, C) ─────────────────────────────────
        h_n_init = x.permute(0, 2, 3, 1).reshape(B, self.n_nodes, -1)  # (B, N, C)

        # ── Derive initial edge features ──────────────────────────────
        # Symmetric component (mean of two endpoints):
        #   x_sum[e] = (1/2) * sum_{v: B1[e,v]≠0} x[v]  = |B₁|ᵀ x / 2
        x_sum_e  = torch.einsum("en, bnc -> bec", self.B1_abs, h_n_init) / 2.0

        # Anti-symmetric component (absolute gradient across the border):
        #   x_diff[e] = | B₁ x |[e]  = | target - source |
        x_diff_e = torch.einsum("en, bnc -> bec", self.B1, h_n_init).abs()

        # Concatenate → (B, E, 2C)
        x_e_init = torch.cat([x_sum_e, x_diff_e], dim=-1)

        # ── Derive initial face features ──────────────────────────────
        # Mean of the four corner node features:
        #   x_face[f] = C_fn @ x_node[f's corners]
        x_f_init = torch.einsum("fn, bnc -> bfc", self.C_fn, h_n_init)  # (B, F, C)

        # ── Embed each dimension to hidden_dim ────────────────────────
        h_n = self.embed_n(h_n_init)   # (B, N, D)
        h_e = self.embed_e(x_e_init)   # (B, E, D)
        h_f = self.embed_f(x_f_init)   # (B, F, D)

        # ── K rounds of CWN message passing ───────────────────────────
        for layer in self.cwn_layers:
            h_n, h_e, h_f = layer(
                h_n, h_e, h_f,
                self.B1_abs, self.B1_abs_T,
                self.B2_abs, self.B2_abs_T,
            )

        # ── Policy head ───────────────────────────────────────────────
        # policy_head is Linear(D → 1) with shared weights across all
        # nodes.  squeeze(-1) removes the trailing 1 dimension.
        policy_logits = self.policy_head(h_n).squeeze(-1)   # (B, N)

        # ── Value head ────────────────────────────────────────────────
        # Sum-pool over all N nodes: D4-invariant global representation.
        # (Sum preserves total material count as a useful signal.)
        v     = h_n.sum(dim=1)               # (B, D)
        value = torch.tanh(self.value_mlp(v)) # (B, 1)

        return policy_logits, value


# ---------------------------------------------------------------------------
# Factory (drop-in replacement for the EGNN build_model)
# ---------------------------------------------------------------------------

def build_model(
    device:        torch.device | None = None,
    compile_model: bool                = False,
) -> tuple[ChainReactionNet, torch.device]:
    """
    Build and return (model, device).

    Drop-in replacement for the original build_model().  All callers
    (main.py, selfplay.py, evaluate.py) use this function unchanged.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChainReactionNet(
        rows        = CFG.rows,
        cols        = CFG.cols,
        in_channels = CFG.num_channels,
        hidden_dim  = CFG.cwn_hidden_dim,
        num_layers  = CFG.cwn_num_layers,
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
