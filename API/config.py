"""
config.py – Central hyperparameter configuration.

All other modules import CFG from here.

CPU-TRAINING TUNING GUIDE
--------------------------
Fastest levers, roughly in order of impact:

  mcts_inference_batch_size
      Leaves per model.forward() call in self-play.
      Set to mcts_simulations for one call per move (maximum batching).

  eval_inference_batch_size
      Same but for evaluation.  Defaults to mcts_simulations so that
      every evaluation move costs exactly ONE forward pass.

  mcts_simulations
      Total MCTS iterations per move.  12 is already aggressive for CPU.

  selfplay_games_per_iter / eval_games
      Both scale linearly with wall time.

  eval_early_stop
      Stop evaluation as soon as the outcome is statistically certain.
      Can save up to 50 % of eval games when results are clear-cut.

  eval_time_limit_s
      Hard wall-clock cap per move during evaluation.  0.05 s is enough
      for relative model comparison.  Set to None to always run all sims.
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
    # own orbs | opp orbs | capacity | own critical | opp critical
    num_channels: int = 5

    # ── MCTS ───────────────────────────────────────────────────────────
    mcts_simulations:   int   = 24
    c_puct:             float = 1.5
    dirichlet_alpha:    float = 0.3
    dirichlet_epsilon:  float = 0.25

    # Number of MCTS leaves evaluated per model.forward() call during
    # self-play.  1 = original single-leaf behaviour (slow on CPU).
    # Set to mcts_simulations to do the entire search in one batch call.
    # Intermediate values (e.g. 8) balance batching vs. tree diversity.
    #
    # Recommended: 8–12 for mcts_simulations ≤ 16
    #              4–8  for mcts_simulations > 32
    mcts_inference_batch_size: int = 4
    # Leaves evaluated per model.forward() call (evaluation).
    # Defaults to mcts_simulations so every eval move is one batch call.
    eval_inference_batch_size: int = 12

    # Stop evaluation early when the final outcome cannot change even if
    # all remaining games go to the other side.  A small margin
    # (eval_early_stop_margin games) is kept to avoid stopping on the
    # exact boundary.  Set eval_early_stop=False to always play all games.
    eval_early_stop: bool = False
    eval_early_stop_margin: int = 3  # safety buffer in games
    eval_min_games: int = 50  # never stop before this many games

    # ── Self-Play ──────────────────────────────────────────────────────
    selfplay_games_per_iter: int = 100
    temperature_threshold:   int = 12
    max_game_length:         int = 70

    # ── Replay Buffer ──────────────────────────────────────────────────
    replay_buffer_size: int = 60_000

    # ── Training ───────────────────────────────────────────────────────
    batch_size:            int   = 256
    train_steps_per_iter:  int   = 100
    learning_rate:         float = 1e-4
    weight_decay:          float = 1e-4
    value_loss_weight:     float = 1.0

    # ── CNN architecture ───────────────────────────────────────────────
    num_filters:    int = 64
    num_res_blocks: int = 4

    # ── Evaluation ─────────────────────────────────────────────────────
    eval_games:           int   = 120
    win_rate_threshold:   float = 0.52
    eval_time_limit_s: Optional[float] = 0.5

    # ── Misc ───────────────────────────────────────────────────────────
    seed:           int = 42
    checkpoint_dir: str = "checkpoints"
    log_interval:   int = 20


CFG = Config()
