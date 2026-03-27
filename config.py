"""
Here we store all important parameters and settings:
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # ── Board-Size ─────────────────────────────────────────────────────
    rows: int = 5
    cols: int = 5

    # ── Encoding ───────────────────────────────────────────────────────
    # own orbs | opp orbs | capacity | own critical | opp critical
    num_channels: int = 5

    # ── MCTS ───────────────────────────────────────────────────────────
    mcts_simulations:   int   = 64
    c_puct:             float = 1.5
    dirichlet_alpha:    float = 0.3
    dirichlet_epsilon:  float = 0.25
    mcts_inference_batch_size: int = 4

    # ── Eval ───────────────────────────────────────────────────────────
    eval_games: int = 150
    win_rate_threshold: float = 0.52
    eval_time_limit_s: Optional[float] = 0.4
    eval_inference_batch_size: int = 12
    eval_early_stop: bool = False
    eval_early_stop_margin: int = 3
    eval_min_games: int = 50

    # ── Self-Play ──────────────────────────────────────────────────────
    selfplay_games_per_iter: int = 150
    temperature_threshold:   int = 10
    max_game_length:         int = 70

    # ── Replay Buffer ──────────────────────────────────────────────────
    replay_buffer_size: int = 60_000

    # ── Training ───────────────────────────────────────────────────────
    batch_size:            int   = 256
    train_steps_per_iter:  int   = 160
    learning_rate:         float = 1e-4
    weight_decay:          float = 1e-4
    value_loss_weight:     float = 1.0

    # ── CNN architecture ───────────────────────────────────────────────
    num_filters:    int = 64
    num_res_blocks: int = 4

    # ── Random ──────────────────────────────────────────────────────────
    seed:           int = 42
    checkpoint_dir: str = "checkpoints"
    log_interval:   int = 20


CFG = Config()
