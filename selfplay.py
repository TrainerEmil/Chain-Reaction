"""
selfplay.py – Generate self-play games and collect training examples.

Each game produces a list of (state, policy_target, value_target) triples.

Value labelling:
  value_target = +1 if the player to move at that position won, else −1.

Temperature schedule:
  Moves < temperature_threshold  →  sample with T=1.0  (exploration)
  Moves ≥ temperature_threshold  →  greedy             (T=0)

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
# Module-level worker state
# ---------------------------------------------------------------------------

_worker_model:  Optional[torch.nn.Module] = None
_worker_device: Optional[torch.device]   = None


def _worker_init(model_state_dict: dict, device_str: str, model_dir: str) -> None:
    """
    Called once per worker process when the pool is created.

    TWO key actions:
      1. Pin PyTorch to 1 thread (see optimisation [2] above).
      2. Load the model into a module-level global (see optimisation [1]).
    """
    import sys
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    global _worker_model, _worker_device

    # ── Thread pinning ────────────────────────────────────────────────
    # Must be called before ANY torch tensor operation in this process.
    # Setting both intra-op and inter-op thread counts to 1 ensures that
    # N worker processes occupy exactly N CPU threads – no more.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # ── Model loading ─────────────────────────────────────────────────
    from model import build_model  # noqa: PLC0415  (avoid circular import in parent)

    _worker_device = torch.device(device_str)

    # compile_model=False in workers: torch.compile's one-time JIT cost
    # (~2–5 s) is amortised over many games in the main process, but for
    # a worker that processes ~3–6 games it is never worth it.
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
    inference_batch_size:  int  = CFG.mcts_inference_batch_size,
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

    history: List = []   # (encoded_state, policy, current_player)
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
            inference_batch_size=inference_batch_size,
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
# Worker task  (lightweight – no model argument)
# ---------------------------------------------------------------------------

def _play_game_worker(
    rows:                  int,
    cols:                  int,
    num_simulations:       int,
    temperature_threshold: int,
    max_game_length:       int,
    inference_batch_size:  int,
    seed: Optional[int],
) -> List[Example]:
    """
    Executed inside a worker process.

    Uses the module-level `_worker_model` / `_worker_device` set up by
    `_worker_init`.  Only small scalar arguments are pickled by the pool.
    """
    return play_game(
        _worker_model,   # type: ignore[arg-type]
        _worker_device,  # type: ignore[arg-type]
        rows=rows,
        cols=cols,
        num_simulations=num_simulations,
        temperature_threshold=temperature_threshold,
        max_game_length=max_game_length,
        inference_batch_size=inference_batch_size,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_selfplay(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int      = CFG.selfplay_games_per_iter,
    seed: Optional[int] = None, model_dir: str = ""
) -> List[Example]:
    """
    Run *num_games* self-play games in parallel and collect examples.

    Parameters
    ----------
    model     : Current best network (weights copied to workers once).
    device    : Torch device (workers always run on CPU regardless).
    num_games : Total games per call.
    seed      : Base seed; game i gets seed+i for reproducibility.

    Returns
    -------
    All (state, policy, value) examples from all completed games.
    """
    model.eval()

    # Serialise model weights once – workers receive them via initialiser.
    model_state_dict = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }

    # One worker per logical CPU, capped at num_games.
    # Each worker uses only 1 PyTorch thread (set in _worker_init), so
    # total threads in flight == min(cpu_count, num_games).
    num_workers = min(os.cpu_count() or 1, num_games)

    all_examples: List[Example] = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(model_state_dict, "cpu", model_dir),
    ) as executor:

        futures = [
            executor.submit(
                _play_game_worker,
                CFG.rows,
                CFG.cols,
                CFG.mcts_simulations,
                CFG.temperature_threshold,
                CFG.max_game_length,
                CFG.mcts_inference_batch_size,
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
