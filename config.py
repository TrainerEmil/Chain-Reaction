"""
config.py – Central hyperparameter configuration.

All other modules import from here so you only need to change
values in one place.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # ── Board ──────────────────────────────────────────────────────────
    rows: int = 5
    cols: int = 5

    # ── Encoding ───────────────────────────────────────────────────────
    # Number of input channels for the CNN (see encoding.py)
    num_channels: int = 5   # own orbs | opp orbs | capacity | own crit | opp crit

    # ── MCTS ───────────────────────────────────────────────────────────
    mcts_simulations: int = 96      # simulations per move (self-play)
    c_puct: float = 1.5             # exploration constant
    dirichlet_alpha: float = 0.3    # noise alpha  (added at root in self-play)
    dirichlet_epsilon: float = 0.25 # weight of Dirichlet noise

    # ── Self-Play ──────────────────────────────────────────────────────
    selfplay_games_per_iter: int = 150   # games generated each iteration
    temperature_threshold: int = 12     # moves < threshold -> sample; else greedy
    max_game_length: int = 120         # safety cap; draw if exceeded

    # ── Replay Buffer ──────────────────────────────────────────────────
    replay_buffer_size: int = 60_000

    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 256
    train_steps_per_iter: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0   # relative weight of value vs policy loss

    # ── CNN architecture ───────────────────────────────────────────────
    num_filters: int = 128
    num_res_blocks: int = 8          # residual blocks in the trunk

    # ── Evaluation ─────────────────────────────────────────────────────
    eval_games: int = 150             # total games per matchup (split P1/P2)
    win_rate_threshold: float = 0.52 # needed to replace the best model

    # Per-move time budget for evaluation MCTS (seconds).
    # MCTS stops adding simulations as soon as this many seconds have
    # elapsed for a single move, regardless of mcts_simulations.
    # Set to None to disable the time cap and always run all simulations.
    #
    # Recommended values:
    #   0.1  – very fast, ~a handful of simulations on a typical CPU
    #   0.5  – balanced: noticeably stronger than random, still quick
    #   1.0  – slow but strong; good for final evaluation
    #   None – no cap; uses mcts_simulations exactly (original behaviour)
    eval_time_limit_s: Optional[float] = 0.2

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 20           # print stats every N train steps


# A global default instance – import this in other modules.
CFG = Config()
