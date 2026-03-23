"""
main.py – AlphaZero training loop for Chain Reaction.

Iteration structure
-------------------
  1. Self-play  → generate training examples with the current best model.
  2. Train      → update the network weights.
  3. Evaluate   → compare new model against random & previous best.
  4. Replace    → update the best model if win-rate threshold is met.
  5. Checkpoint → persist the full training state to disk.

Resume support
--------------
After every iteration a complete training-state snapshot is written to
  <checkpoint_dir>/training_state.pt

This file contains:
  - model weights          (current training model)
  - best_model weights     (current accepted best)
  - optimizer state        (momentum, adaptive LR, etc.)
  - replay buffer contents (all stored examples)
  - last completed iteration number

On the next run the snapshot is detected automatically and training
continues from where it left off.  Use --reset to start fresh.

Usage
-----
  python main.py                        # auto-resume or start fresh
  python main.py --iterations 40        # run up to iteration 40 total
  python main.py --reset                # ignore any checkpoint, start fresh
  python main.py --reset --iterations 10
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

from config import CFG
from evaluate import evaluate_vs_model, evaluate_vs_random
from model import build_model, ChainReactionNet
from replay_buffer import ReplayBuffer
from selfplay import run_selfplay
from train import build_optimizer, train


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

# The single file that holds the *full* resumable training state.
_STATE_FILE = "training_state.pt"


def state_path() -> str:
    return os.path.join(CFG.checkpoint_dir, _STATE_FILE)


def save_training_state(
    iteration: int,
    model: ChainReactionNet,
    best_model: ChainReactionNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
) -> None:
    """
    Persist the complete training state after each iteration so that
    training can be resumed exactly where it stopped.

    Saved fields
    ------------
    iteration        : Last fully completed iteration (int).
    model_state      : state_dict of the current training model.
    best_model_state : state_dict of the current accepted best model.
    optimizer_state  : Full optimizer state (momentum buffers, etc.).
    buffer_data      : List of (state, policy, value) tuples from the
                       replay buffer, in insertion order.
    """
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    # Serialise the replay buffer as a plain list so it does not depend
    # on the ReplayBuffer class internals when loading.
    buffer_data = list(buffer._buffer)

    payload = {
        "iteration":        iteration,
        "model_state":      model.state_dict(),
        "best_model_state": best_model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "buffer_data":      buffer_data,
    }

    # Write to a temporary file first, then rename atomically.
    # This prevents a corrupted checkpoint if the process is killed
    # mid-write.
    tmp_path = state_path() + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, state_path())

    print(f"  ✓ Training state saved → {state_path()}  "
          f"(iter={iteration}, buffer={len(buffer_data)} examples)")


def load_training_state(
    model: ChainReactionNet,
    best_model: ChainReactionNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
) -> int:
    """
    Load the training state from disk and restore all objects in-place.

    Returns
    -------
    int : The last completed iteration number.
          Training should resume from iteration + 1.
    """
    path = state_path()
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model_state"])
    best_model.load_state_dict(payload["best_model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])

    # Restore the replay buffer contents preserving insertion order.
    buffer._buffer = deque(payload["buffer_data"], maxlen=buffer.max_size)

    iteration = payload["iteration"]
    print(f"  ✓ Training state loaded ← {path}  "
          f"(resuming after iter={iteration}, buffer={len(buffer)} examples)")
    return iteration


def save_model_snapshot(model: ChainReactionNet, iteration: int) -> None:
    """
    Save a standalone model-weights file for this iteration.
    These files are kept for later evaluation / rollback and are
    separate from the resumable training state.
    """
    path = os.path.join(CFG.checkpoint_dir, f"model_iter_{iteration:03d}.pt")
    torch.save(model.state_dict(), path)
    print(f"  ✓ Model snapshot saved  → {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(num_iterations: int = 20, reset: bool = False) -> None:
    set_seeds(CFG.seed)
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    # ── Build model, optimizer, buffer ────────────────────────────────
    model,      device = build_model()
    best_model, _      = build_model(device)
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))

    optimizer = build_optimizer(model)
    buffer    = ReplayBuffer(CFG.replay_buffer_size)

    # ── Resume or start fresh ─────────────────────────────────────────
    start_iteration = 1

    if not reset and os.path.exists(state_path()):
        print("\n  Found existing training state – resuming …")
        completed = load_training_state(model, best_model, optimizer, buffer, device)
        start_iteration = completed + 1
    else:
        if reset:
            print("\n  --reset flag set – starting fresh (ignoring any checkpoint).")
        else:
            print("\n  No checkpoint found – starting from scratch.")

    end_iteration = max(start_iteration, num_iterations)
    remaining     = end_iteration - start_iteration + 1

    if remaining <= 0:
        print(f"\n  All {num_iterations} iterations already completed. Nothing to do.")
        print(f"  To train further, run:  python main.py --iterations {num_iterations + 10}")
        return

    print(f"\nTraining on device : {device}")
    print(f"Board              : {CFG.rows}×{CFG.cols}")
    print(f"MCTS simulations   : {CFG.mcts_simulations}")
    print(f"Iterations         : {start_iteration} → {end_iteration}  ({remaining} to run)")
    print("=" * 60)

    # ── Iteration loop ────────────────────────────────────────────────
    for iteration in range(start_iteration, end_iteration + 1):
        t0 = time.time()
        print(f"\n── Iteration {iteration}/{end_iteration} ──────────────────────")

        # 1. Self-play ─────────────────────────────────────────────────
        print(f"[1] Self-play ({CFG.selfplay_games_per_iter} games) …")
        examples = run_selfplay(
            best_model, device,
            num_games=CFG.selfplay_games_per_iter,
            seed=CFG.seed + iteration,   # deterministic per-iteration seed
        )
        buffer.add(examples)
        print(f"    Buffer size: {len(buffer)}")

        # 2. Train ─────────────────────────────────────────────────────
        print(f"[2] Training ({CFG.train_steps_per_iter} steps) …")
        train(model, optimizer, buffer, device)

        # 3. Evaluate ──────────────────────────────────────────────────
        print("[3] Evaluation …")
        evaluate_vs_random(model, device, num_games=CFG.eval_games)

        replace = False
        if iteration > 1:
            result = evaluate_vs_model(
                model, best_model, device, num_games=CFG.eval_games
            )
            replace = result["win_rate"] >= CFG.win_rate_threshold
        else:
            replace = True   # always accept the very first model

        # 4. Maybe replace best model ──────────────────────────────────
        if replace:
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
            save_model_snapshot(best_model, iteration)
            print(f"  ✓ New best model accepted (iter {iteration}).")
        else:
            print(f"  ✗ Model not accepted – reverting to previous best.")
            model.load_state_dict(copy.deepcopy(best_model.state_dict()))
            optimizer = build_optimizer(model)   # reset optimizer momentum

        elapsed = time.time() - t0
        print(f"  Iteration time: {elapsed:.1f}s")

        # 5. Save full training state ──────────────────────────────────
        save_training_state(iteration, model, best_model, optimizer, buffer)

    # ── Done ──────────────────────────────────────────────────────────
    final_path = os.path.join(CFG.checkpoint_dir, "model_final.pt")
    torch.save(best_model.state_dict(), final_path)
    print(f"\n{'=' * 60}")
    print(f"Training complete.  Best model → {final_path}")
    print(f"To continue training run:  python main.py --iterations {end_iteration + 10}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain Reaction – AlphaZero training")
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Total number of iterations to run (default: 20).",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Ignore any existing checkpoint and start training from scratch.",
    )
    args = parser.parse_args()
    main(num_iterations=args.iterations, reset=args.reset)
