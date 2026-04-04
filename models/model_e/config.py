"""
config.py – All hyperparameters and architectural settings.

CR-CWN specific additions
--------------------------
  cwn_hidden_dim : feature vector width at every CWN layer (nodes, edges, faces)
  cwn_num_layers : number of CWN rounds (each round = 4-way message pass)

Derived channel counts (informational; do not set directly):
  node  channels = num_channels           (= 5)
  edge  channels = 2 * num_channels       (= 10:  mean + |gradient| of endpoints)
  face  channels = num_channels           (= 5:   mean of 4 corner nodes)
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
    mcts_simulations:          int   = 32
    c_puct:                    float = 1.5
    dirichlet_alpha:           float = 0.3
    dirichlet_epsilon:         float = 0.25
    mcts_inference_batch_size: int   = 4

    # ── Evaluation ─────────────────────────────────────────────────────
    eval_games:               int            = 100
    win_rate_threshold:       float          = 0.52
    eval_time_limit_s:        Optional[float] = 0.1
    eval_inference_batch_size: int           = 12
    eval_early_stop:          bool           = False
    eval_early_stop_margin:   int            = 3
    eval_min_games:           int            = 50

    # ── Self-play ──────────────────────────────────────────────────────
    selfplay_games_per_iter: int = 100
    temperature_threshold:   int = 10
    max_game_length:         int = 70

    # ── Replay buffer ──────────────────────────────────────────────────
    replay_buffer_size: int = 60_000

    # ── Training ───────────────────────────────────────────────────────
    batch_size:           int   = 256
    train_steps_per_iter: int   = 200
    learning_rate:        float = 1e-4
    weight_decay:         float = 1e-4
    value_loss_weight:    float = 1.0

    # ── CR-CWN architecture ────────────────────────────────────────────
    # hidden_dim : width of node/edge/face feature vectors in every CWN layer.
    # num_layers : number of CWN message-passing rounds.  At round k the network
    #              can integrate information from cells that are k hops away at
    #              any dimension (node, edge, or face), giving an effective
    #              receptive field that grows across all three cell levels.
    cwn_hidden_dim: int = 48
    cwn_num_layers: int = 4

    # ── Legacy GNN fields (kept for checkpoint compatibility) ───────────
    # These fields are no longer used by CR-CWN but are retained so that
    # old training-state files can still be loaded without KeyErrors.
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 6
    num_filters:    int = 64
    num_res_blocks: int = 4

    # ── Misc ───────────────────────────────────────────────────────────
    seed:           int = 42
    checkpoint_dir: str = str(Path(__file__).parent / "checkpoints")
    log_interval:   int = 20


CFG = Config()
