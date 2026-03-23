"""
encoding.py – Converts a ChainReaction game state into a tensor that the
CNN can consume, and builds the legal-action mask.

Channels (from the current player's perspective):
  0 : own orbs       – normalised count of orbs the current player owns
  1 : opponent orbs  – normalised count of opponent orbs
  2 : capacity       – capacity of each cell (normalised, static)
  3 : own critical   – cells where own orbs == cap - 1 (one away from exploding)
  4 : opp critical   – cells where opp orbs == cap - 1

All values are floats in [0, 1].

The key design decision is *perspective encoding*: the network always
sees the board as "me vs. them", regardless of which physical player is
acting.  This allows the same network weights to be used for both players
and simplifies training.
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
    n = game.n
    s = game.current_player          # +1 for P1, -1 for P2
    board = game.board
    cap = game.cap

    # Maximum possible orbs in a cell – used for normalisation.
    # A cell can theoretically accumulate many orbs, but in practice
    # values above cap are rare.  We normalise by max capacity (4) to
    # keep values in a sensible range; values > 1 are fine for the net.
    max_cap = 4.0

    own_orbs  = np.zeros((rows, cols), dtype=np.float32)
    opp_orbs  = np.zeros((rows, cols), dtype=np.float32)
    capacity  = np.zeros((rows, cols), dtype=np.float32)
    own_crit  = np.zeros((rows, cols), dtype=np.float32)
    opp_crit  = np.zeros((rows, cols), dtype=np.float32)

    for i in range(n):
        r, c = divmod(i, cols)
        v = board[i]
        ci = cap[i]

        capacity[r, c] = ci / max_cap

        if v == 0:
            continue

        orbs = abs(v)
        owner_sign = 1 if v > 0 else -1

        if owner_sign == s:          # current player owns this cell
            own_orbs[r, c] = orbs / max_cap
            if orbs == ci - 1:
                own_crit[r, c] = 1.0
        else:                        # opponent owns this cell
            opp_orbs[r, c] = orbs / max_cap
            if orbs == ci - 1:
                opp_crit[r, c] = 1.0

    # Stack into shape (5, rows, cols)
    return np.stack([own_orbs, opp_orbs, capacity, own_crit, opp_crit], axis=0)


def legal_action_mask(game: ChainReaction) -> np.ndarray:
    """
    Return a boolean mask of shape (rows * cols,) where True means legal.

    The network's policy head outputs logits over all n cells.
    We zero-out illegal cells before the softmax during MCTS and training.
    """
    n = game.n
    s = game.current_player
    board = game.board
    mask = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if board[i] == 0 or board[i] * s > 0:
            mask[i] = 1.0
    return mask


def encode_state_tensor(game: ChainReaction) -> torch.Tensor:
    """Convenience wrapper: encode and return a (1, C, H, W) float32 tensor."""
    arr = encode_state(game)
    return torch.from_numpy(arr).unsqueeze(0)   # add batch dim
