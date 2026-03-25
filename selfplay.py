"""
selfplay.py – Generate self-play games and collect training examples.

Each game produces a list of (state, policy_target, value_target) triples.

Value labelling:
  After the game ends we know the winner.  For every stored position we set
  value_target = +1 if the player to move at that position won, else -1.
  Consistent with the network's perspective encoding and MCTS backprop.

Temperature schedule:
  Moves < temperature_threshold  →  sample with T=1.0  (exploration)
  Moves ≥ temperature_threshold  →  greedy             (T=0)

OPTIMISATIONS IN THIS FILE
--------------------------
Worker-pool initialiser (biggest gain)
    Original: the model state-dict was passed as a function *argument* to
    every submitted task, so it was pickled once per game (50× per
    iteration for CFG.selfplay_games_per_iter=50).  For a model with
    ~1–2 M parameters that is ~4–8 MB of IPC data per game.

    Fixed: ProcessPoolExecutor is created with an `initializer` that loads
    the model state-dict into a module-level global *once per worker
    process*.  The worker function receives only lightweight scalar
    arguments (ints + optional seed).  IPC overhead drops from O(games)
    to O(workers) – typically 8–16× less data sent over the pipe.

Worker-count heuristic
    CPU-bound tasks benefit most from one process per *physical* core.
    `os.cpu_count()` returns logical cores (including HyperThreading
    siblings), which can double the apparent count.  We cap workers at
    `min(num_games, os.cpu_count())` but the user can override via
    CFG.num_selfplay_workers if they want tighter control.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import torch

from config import CFG
from encoding import encode_state
from engine import ChainReaction
from mcts import MCTS
from replay_buffer import Example


# ---------------------------------------------------------------------------
# Module-level worker state  (populated by _worker_init, used by workers)
# ---------------------------------------------------------------------------

_worker_model: Optional[torch.nn.Module] = None
_worker_device: Optional[torch.device]   = None

def _worker_init(model_state_dict: dict, device_str: str) -> None:
    """
    Initialiser called once per worker process when the pool starts.

    Loads the model into a module-level global so that subsequent tasks
    submitted to this worker do not need to re-deserialise the state dict.

    Cost: O(workers), not O(games).
    """
    global _worker_model, _worker_device

    # Import here so child processes don't inherit the parent's CUDA context
    from model import build_model  # noqa: PLC0415

    _worker_device = torch.device(device_str)
    # compile_model=False in workers: compilation overhead is not worth it
    # for short-lived worker processes.
    _worker_model, _ = build_model(_worker_device, compile_model=False)
    _worker_model.load_state_dict(model_state_dict)
    _worker_model.eval()


# ---------------------------------------------------------------------------
# Game-play logic
# ---------------------------------------------------------------------------

def play_game(
    model: torch.nn.Module,
    device: torch.device,
    rows:                  int  = CFG.rows,
    cols:                  int  = CFG.cols,
    num_simulations:       int  = CFG.mcts_simulations,
    temperature_threshold: int  = CFG.temperature_threshold,
    max_game_length:       int  = CFG.max_game_length,
    seed: Optional[int]         = None,
) -> List[Example]:
    """
    Play one complete self-play game and return training examples.

    Returns an empty list if the game reaches max_game_length (draw).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    mcts = MCTS(model, device)
    mcts.reset()

    game = ChainReaction(rows, cols)
    game.reset()

    history = []          # (encoded_state, policy, current_player)
    move_number = 0
    winner      = None

    while winner is None and move_number < max_game_length:
        temperature = 1.0 if move_number < temperature_threshold else 0.0

        encoded = encode_state(game)
        player  = game.current_player

        policy, action = mcts.run(
            game,
            num_simulations=num_simulations,
            temperature=temperature,
            add_noise=True,
            reuse_tree=True,
        )

        history.append((encoded, policy, player))
        winner = game.step(action)
        mcts.advance_to_action(action, game)
        move_number += 1

    if winner is None:
        return []   # draw – no usable training signal

    return [
        (encoded, policy, 1.0 if player == winner else -1.0)
        for encoded, policy, player in history
    ]


# ---------------------------------------------------------------------------
# Lightweight worker task  (no model argument – uses the global)
# ---------------------------------------------------------------------------

def _play_game_worker(
    rows:                  int,
    cols:                  int,
    num_simulations:       int,
    temperature_threshold: int,
    max_game_length:       int,
    seed: Optional[int],
) -> List[Example]:
    """
    Task executed inside a worker process.

    Uses the module-level `_worker_model` and `_worker_device` that were
    set up once by `_worker_init`.  No model state dict is transferred;
    only small scalar arguments are pickled by the pool.
    """
    return play_game(
        _worker_model,   # type: ignore[arg-type]
        _worker_device,  # type: ignore[arg-type]
        rows=rows,
        cols=cols,
        num_simulations=num_simulations,
        temperature_threshold=temperature_threshold,
        max_game_length=max_game_length,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_selfplay(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int           = CFG.selfplay_games_per_iter,
    seed: Optional[int]      = None,
) -> List[Example]:
    """
    Run *num_games* self-play games in parallel and aggregate examples.

    Parameters
    ----------
    model     : Current best network (weights are copied to workers).
    device    : Torch device (only CPU workers are spawned regardless).
    num_games : Number of games to play.
    seed      : Base seed; game i gets seed+i for reproducibility.

    Returns
    -------
    All (state, policy, value) examples from all completed games.
    """
    model.eval()

    # Serialise weights once (not once per game)
    model_state_dict = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }

    # Number of workers: at most one per game, and at most cpu_count.
    # We avoid spawning more workers than physical tasks to prevent
    # excessive process-creation overhead.
    num_workers = min(os.cpu_count() or 1, num_games)

    all_examples: List[Example] = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(model_state_dict, "cpu"),   # workers always use CPU
    ) as executor:
        futures = [
            executor.submit(
                _play_game_worker,
                CFG.rows,
                CFG.cols,
                CFG.mcts_simulations,
                CFG.temperature_threshold,
                CFG.max_game_length,
                None if seed is None else seed + i,
            )
            for i in range(num_games)
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            examples = future.result()
            all_examples.extend(examples)

            if i % 10 == 0 or i == num_games:
                print(
                    f"  Self-play: {i}/{num_games} games done, "
                    f"{len(all_examples)} examples so far."
                )

    return all_examples
