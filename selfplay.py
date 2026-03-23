"""
selfplay.py – Generate self-play games and collect training examples.

Each game produces a list of (state, policy_target, value_target) triples.

Value labelling:
  After the game ends we know the winner.  For every stored position we set
  value_target = +1 if the player to move at that position won, else -1.
  This means value_target is always from the current-player's perspective,
  consistent with how the network is trained and how MCTS backpropagates.

Temperature schedule:
  Moves < temperature_threshold → sample with T=1.0 (exploration).
  Moves >= temperature_threshold → greedy (T=0).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from config import CFG
from encoding import encode_state
from engine import ChainReaction, P1, P2
from mcts import MCTS
from replay_buffer import Example


def play_game(
    model: torch.nn.Module,
    device: torch.device,
    rows: int = CFG.rows,
    cols: int = CFG.cols,
    num_simulations: int = CFG.mcts_simulations,
    temperature_threshold: int = CFG.temperature_threshold,
    max_game_length: int = CFG.max_game_length,
    seed: Optional[int] = None,
) -> List[Example]:
    """
    Play a complete self-play game and return training examples.

    Parameters
    ----------
    model               : The current best network (used by MCTS).
    device              : Torch device.
    rows, cols          : Board size.
    num_simulations     : MCTS iterations per move.
    temperature_threshold : Moves below this index use T=1, above use T=0.
    max_game_length     : Game is declared a draw if it exceeds this.
    seed                : Optional RNG seed for reproducibility.

    Returns
    -------
    List of (encoded_state, policy_target, value_target) for every move.
    Returns an empty list if the game was a draw (no training signal).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    mcts = MCTS(model, device)
    game = ChainReaction(rows, cols)
    game.reset()

    # Intermediate storage: (encoded_state, policy_target, player_at_move)
    history: List[Tuple[np.ndarray, np.ndarray, int]] = []

    move_number = 0
    winner: Optional[int] = None

    while winner is None and move_number < max_game_length:
        temperature = 1.0 if move_number < temperature_threshold else 0.0
        add_noise = True   # always explore during self-play

        # Encode state *before* the move (from current player's perspective)
        encoded = encode_state(game)
        player = game.current_player

        policy, action = mcts.run(
            game,
            num_simulations=num_simulations,
            temperature=temperature,
            add_noise=add_noise,
        )

        history.append((encoded, policy, player))
        winner = game.step(action)
        move_number += 1

    if winner is None:
        # Draw – no meaningful value signal; discard this game.
        return []

    # Assign value targets: +1 if that position's player won, else -1.
    examples: List[Example] = []
    for encoded, policy, player in history:
        value_target = 1.0 if player == winner else -1.0
        examples.append((encoded, policy, value_target))

    return examples


def run_selfplay(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int = CFG.selfplay_games_per_iter,
    seed: Optional[int] = None,
) -> List[Example]:
    """
    Run *num_games* self-play games and aggregate all training examples.

    Parameters
    ----------
    model     : Network to use for MCTS.
    device    : Torch device.
    num_games : Number of games to play.
    seed      : Base seed; each game gets seed + game_index for reproducibility.

    Returns
    -------
    All (state, policy, value) examples from all games.
    """
    model.eval()
    all_examples: List[Example] = []

    for i in range(num_games):
        game_seed = None if seed is None else seed + i
        examples = play_game(model, device, seed=game_seed)
        all_examples.extend(examples)

        if (i + 1) % 10 == 0 or (i + 1) == num_games:
            print(f"  Self-play: {i + 1}/{num_games} games done, "
                  f"{len(all_examples)} examples so far.")

    return all_examples
