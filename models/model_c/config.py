"""
config.py – Central hyperparameter configuration.

All other modules import CFG from here.

CPU-TRAINING NOTES
------------------
The defaults below are tuned for a modern multi-core CPU (8–16 cores).
Key knobs for iteration speed:

  mcts_simulations       – most direct lever; fewer sims = faster self-play.
                           12 is already aggressive; do not go below 8 or
                           the MCTS policy signal degrades significantly.

  selfplay_games_per_iter – scales linearly with wall time.  Reduce early
                           in training when the buffer fills up quickly anyway.

  eval_time_limit_s       – bounds evaluation cost.  0.05 s/move means MCTS
                           runs until the timer fires (often just 1–3 sims
                           on a slow machine), which is fine for relative
                           comparison between models.

  num_filters / num_res_blocks – smaller networks are faster but weaker.
                           The defaults (64 filters, 4 blocks) are a good
                           trade-off for CPU.  Increase after training is
                           stable.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    # ── Board ──────────────────────────────────────────────────────────
    rows: int = 5
    cols: int = 5

    # ── Encoding ───────────────────────────────────────────────────────
    # own orbs | opp orbs | capacity | own critical | opp critical
    num_channels: int = 5

    # ── MCTS ───────────────────────────────────────────────────────────
    mcts_simulations:   int   = 16
    c_puct:             float = 1.5
    dirichlet_alpha:    float = 0.3
    dirichlet_epsilon:  float = 0.25
    mcts_inference_batch_size: int = 4

    # ── Self-Play ──────────────────────────────────────────────────────
    selfplay_games_per_iter: int = 50
    temperature_threshold:   int = 12
    max_game_length:         int = 70
    num_selfplay_workers: Optional[int] = None

    # ── Replay Buffer ──────────────────────────────────────────────────
    replay_buffer_size: int = 10_000

    # ── Training ───────────────────────────────────────────────────────
    batch_size:            int   = 256
    train_steps_per_iter:  int   = 100
    learning_rate:         float = 1e-4
    weight_decay:          float = 1e-4
    value_loss_weight:     float = 1.0

    # ── CNN architecture ───────────────────────────────────────────────
    # These control the backbone shared by all 8 D4 copies in one batch.
    # Larger values are safe (the optimised forward pass handles bigger
    # batch sizes efficiently), but increase per-step training time.
    num_filters:    int = 64
    num_res_blocks: int = 4

    # ── Evaluation ─────────────────────────────────────────────────────
    eval_games:           int   = 50
    win_rate_threshold:   float = 0.52

    # Per-move time budget for evaluation MCTS (seconds).
    # 0.05 s → usually 1–5 MCTS sims on a typical CPU, very fast.
    # None  → use mcts_simulations exactly (slow on CPU).
    eval_time_limit_s: Optional[float] = 0.1
    eval_inference_batch_size: int = 12
    eval_early_stop: bool = False
    eval_early_stop_margin: int = 3
    eval_min_games: int = 50

    # ── Misc ───────────────────────────────────────────────────────────
    seed:           int = 42
    checkpoint_dir: str = str(Path(__file__).parent / "checkpoints")
    log_interval:   int = 20


# Global default instance imported by all other modules.
CFG = Config()
