"""
encoding.py – Board state → tensor encoding, plus cell-complex utilities.

Core functions (unchanged from the EGNN version)
-------------------------------------------------
  encode_state(game)         → (5, rows, cols) float32 numpy array
  legal_action_mask(game)    → (rows*cols,) float32 numpy array
  encode_state_tensor(game)  → (1, 5, rows, cols) torch.Tensor

  These are the functions called by mcts.py, selfplay.py, and train.py.
  Their interface and output are identical to the previous version.

Cell-complex utility functions (new; used for analysis and documentation)
------------------------------------------------------------------------
  encode_edge_features(game) → (E, 2*C_n) float32 numpy array
  encode_face_features(game) → (F, C_n)   float32 numpy array

  Note: the CR-CWN model derives edge and face features *internally*
  from the node feature tensor during its forward pass (using registered
  topology buffers B₁ and C_fn).  These two functions expose the same
  computation as standalone utilities, e.g. for visualisation or sanity
  checks.  They are NOT called by the model or any training loop.

5-channel node encoding
-----------------------
  ch 0 : own orbs       – normalised count owned by the current player
  ch 1 : opponent orbs  – normalised count owned by the opponent
  ch 2 : capacity       – cell capacity (static; normalised)
  ch 3 : own critical   – 1.0 if own orbs == cap − 1, else 0.0
  ch 4 : opp critical   – 1.0 if opp orbs == cap − 1, else 0.0

Edge features (C_e = 10)
------------------------
  ch 0..4  : (node_feat[u] + node_feat[v]) / 2     symmetric mean
  ch 5..9  : |node_feat[u] − node_feat[v]|         absolute gradient

Face features (C_f = 5)
-----------------------
  ch 0..4  : mean of 4 corner node features
"""

import numpy as np
import torch

from engine import ChainReaction, P1


# ---------------------------------------------------------------------------
# Core encoding (unchanged)
# ---------------------------------------------------------------------------

def encode_state(game: ChainReaction) -> np.ndarray:
    """
    Encode the current game state as a float32 numpy array.

    Returns
    -------
    np.ndarray, shape (5, rows, cols), dtype float32
    """
    rows, cols = game.rows, game.cols
    s = game.current_player          # +1 for P1, -1 for P2
    max_cap = 4.0

    board = np.asarray(game.board, dtype=np.int32).reshape(rows, cols)
    cap   = np.asarray(game.cap,   dtype=np.float32).reshape(rows, cols)

    orbs = np.abs(board).astype(np.float32)

    own_mask = (board * s) > 0
    opp_mask = (board * s) < 0

    own_orbs = np.where(own_mask, orbs, 0.0) / max_cap
    opp_orbs = np.where(opp_mask, orbs, 0.0) / max_cap

    capacity = cap / max_cap

    cap_minus_1 = cap - 1.0
    own_crit = np.where(own_mask & (orbs == cap_minus_1), 1.0, 0.0).astype(np.float32)
    opp_crit = np.where(opp_mask & (orbs == cap_minus_1), 1.0, 0.0).astype(np.float32)

    return np.ascontiguousarray(
        np.stack([own_orbs, opp_orbs, capacity, own_crit, opp_crit]).astype(np.float32)
    )


def legal_action_mask(game: ChainReaction) -> np.ndarray:
    """
    Return a float32 mask of shape (rows * cols,).
    1.0 = legal move, 0.0 = illegal.
    """
    s = game.current_player
    board = np.asarray(game.board, dtype=np.int32)
    return ((board == 0) | (board * s > 0)).astype(np.float32)


def encode_state_tensor(game: ChainReaction) -> torch.Tensor:
    """Convenience wrapper: encode and return a (1, C, H, W) float32 tensor."""
    arr = encode_state(game)
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Cell-complex utilities (new; for analysis only – not called by the model)
# ---------------------------------------------------------------------------

def encode_edge_features(game: ChainReaction) -> np.ndarray:
    """
    Encode edge features for the current game state.

    For each edge e = (u, v) the feature vector is:
      channels 0..C-1  :  (node_feat[u] + node_feat[v]) / 2    (mean)
      channels C..2C-1 :  |node_feat[u] − node_feat[v]|         (|gradient|)

    Edge ordering matches build_boundary_operators() in model.py:
      horizontal edges first  (rows × (cols−1) edges),
      then vertical edges     ((rows−1) × cols edges).

    Returns
    -------
    np.ndarray, shape (E, 2 * num_channels), dtype float32
    where E = rows*(cols-1) + (rows-1)*cols.
    """
    rows, cols = game.rows, game.cols

    # (num_channels, rows, cols) → (N, num_channels)
    node_feat = encode_state(game)
    x = node_feat.reshape(node_feat.shape[0], rows * cols).T  # (N, C)

    means, grads = [], []

    # Horizontal edges: (r, c) → (r, c+1)
    for r in range(rows):
        for c in range(cols - 1):
            u = r * cols + c
            v = r * cols + c + 1
            means.append((x[u] + x[v]) / 2.0)
            grads.append(np.abs(x[u] - x[v]))

    # Vertical edges: (r, c) → (r+1, c)
    for r in range(rows - 1):
        for c in range(cols):
            u = r * cols + c
            v = (r + 1) * cols + c
            means.append((x[u] + x[v]) / 2.0)
            grads.append(np.abs(x[u] - x[v]))

    return np.concatenate([
        np.stack(means, axis=0),   # (E, C)
        np.stack(grads, axis=0),   # (E, C)
    ], axis=1).astype(np.float32)  # (E, 2C)


def encode_face_features(game: ChainReaction) -> np.ndarray:
    """
    Encode face (plaquette) features for the current game state.

    For each 2×2 plaquette f the feature vector is the mean of the four
    corner node features:

      face_feat[f] = mean( node_feat[c] for c in corners(f) )

    Face ordering: row-major, plaquette (r, c) has index r*(cols−1) + c.

    Returns
    -------
    np.ndarray, shape (F, num_channels), dtype float32
    where F = (rows-1)*(cols-1).
    """
    rows, cols = game.rows, game.cols
    C_n = 5   # number of node channels

    node_feat = encode_state(game)
    x = node_feat.reshape(C_n, rows * cols).T   # (N, C_n)

    F = (rows - 1) * (cols - 1)
    face_feat = np.zeros((F, C_n), dtype=np.float32)

    for r in range(rows - 1):
        for c in range(cols - 1):
            f = r * (cols - 1) + c
            corners = [
                r       * cols + c,
                r       * cols + c + 1,
                (r + 1) * cols + c,
                (r + 1) * cols + c + 1,
            ]
            face_feat[f] = x[corners].mean(axis=0)

    return face_feat
