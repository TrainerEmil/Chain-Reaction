"""
Here we convert a ChainReaction game state into a tensor which the
CNN can consume, and builds the legal-action mask.

We use the following 5 channels
  0 : own orbs       – normalised count of orbs the current player owns
  1 : opponent orbs  – normalised count of opponent orbs
  2 : capacity       – capacity of each cell (normalised, static)
  3 : own critical   – cells where own orbs == cap - 1
  4 : opp critical   – cells where opp orbs == cap - 1

All values are floats in [0, 1].

"""

import numpy as np
import torch

from engine import ChainReaction, P1


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

    # Convert Python lists → NumPy arrays once (O(n) in C, not Python)
    board = np.asarray(game.board, dtype=np.int32).reshape(rows, cols)
    cap   = np.asarray(game.cap,   dtype=np.float32).reshape(rows, cols)

    orbs = np.abs(board).astype(np.float32)   # unsigned orb count per cell

    # Boolean masks – no Python branching per cell
    own_mask = (board * s) > 0   # cells that belong to the current player
    opp_mask = (board * s) < 0   # cells that belong to the opponent

    # Channel 0 & 1: normalised orb counts per player
    own_orbs = np.where(own_mask, orbs, 0.0) / max_cap
    opp_orbs = np.where(opp_mask, orbs, 0.0) / max_cap

    # Channel 2: cell capacity (static, same for both players)
    capacity = cap / max_cap

    # Channel 3 & 4: "critical" cells (one explosion away)
    # A cell is critical when  orbs == cap - 1
    cap_minus_1 = cap - 1.0
    own_crit = np.where(own_mask & (orbs == cap_minus_1), 1.0, 0.0).astype(np.float32)
    opp_crit = np.where(opp_mask & (orbs == cap_minus_1), 1.0, 0.0).astype(np.float32)

    # Stack → (5, rows, cols) and ensure a contiguous memory layout
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
    # A cell is legal when it is empty OR owned by the current player
    return ((board == 0) | (board * s > 0)).astype(np.float32)


def encode_state_tensor(game: ChainReaction) -> torch.Tensor:
    """Convenience wrapper: encode and return a (1, C, H, W) float32 tensor."""
    arr = encode_state(game)
    # from_numpy shares memory with arr (zero-copy); unsqueeze adds batch dim.
    return torch.from_numpy(arr).unsqueeze(0)
