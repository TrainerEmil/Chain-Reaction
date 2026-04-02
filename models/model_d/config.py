from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    rows:    int = 5
    cols:    int = 5
    num_channels: int = 5

    mcts_simulations:          int   = 16
    c_puct:                    float = 1.5
    dirichlet_alpha:           float = 0.3
    dirichlet_epsilon:         float = 0.25
    mcts_inference_batch_size: int   = 4

    eval_games:                int            = 50
    win_rate_threshold:        float          = 0.52
    eval_time_limit_s:         Optional[float] = 0.2
    eval_inference_batch_size: int            = 12
    eval_early_stop:           bool           = False
    eval_early_stop_margin:    int            = 3
    eval_min_games:            int            = 50

    selfplay_games_per_iter:   int = 50
    temperature_threshold:     int = 10
    max_game_length:           int = 70

    replay_buffer_size:        int = 10_000
    batch_size:                int = 256
    train_steps_per_iter:      int = 100
    learning_rate:             float = 1e-4
    weight_decay:              float = 1e-4
    value_loss_weight:         float = 1.0

    num_filters:               int = 64
    num_res_blocks:            int = 4
    gnn_hidden_dim:            int = 128
    gnn_num_layers:            int = 6

    seed:           int = 42
    checkpoint_dir: str = str(Path(__file__).parent / "checkpoints")
    log_interval:   int = 20


CFG = Config()