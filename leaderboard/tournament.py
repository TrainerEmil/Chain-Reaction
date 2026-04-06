#!/usr/bin/env python3
"""
leaderboard/tournament.py  –  Round-robin tournament with persistent Elo ratings.
==================================================================================

Each model runs in its OWN subprocess with its own sys.path, so models with
completely different encodings, configs, and architectures can compete fairly.

Persistent Elo
--------------
Ratings are stored in  leaderboard/results/elo_ratings.json  and updated after
every match.  New models start at 1000.  Run the tournament repeatedly and
ratings accumulate across sessions.

Round-robin
-----------
One ROUND = every pair of models plays one match.  Each match uses N random
starting positions (--positions, default 5).  Every position is played twice
with the colour assignment swapped, so total games per match = positions × 2.
Use --rounds N to repeat the full round-robin N times.
Elo is updated after each individual match so the leaderboard evolves live.

Usage
-----
  # Auto-discover all models in models/, 1 round, 5 positions per match
  python leaderboard/tournament.py

  # 3 rounds
  python leaderboard/tournament.py --rounds 3

  # Only specific models
  python leaderboard/tournament.py --models model_b model_c

  # More positions and longer think time
  python leaderboard/tournament.py --positions 10 --time-limit 0.5

  # Control the opening length (0 = always start empty, 5 = 0–5 random moves)
  python leaderboard/tournament.py --max-opening 3

  # More MCTS simulations
  python leaderboard/tournament.py --simulations 32

Full example:
  python leaderboard/tournament.py --rounds 5 --positions 10 --time-limit 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random as _random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# ── Project root (one level up from leaderboard/) ─────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # engine.py always lives at the root

from engine import ChainReaction, P1, P2

# ── Persistent storage paths ───────────────────────────────────────────────
RESULTS_DIR  = ROOT / "leaderboard" / "results"
RATINGS_FILE = RESULTS_DIR / "elo_ratings.json"
HISTORY_DIR  = RESULTS_DIR / "history"

# ── Elo constants ──────────────────────────────────────────────────────────
ELO_START = 1000.0
ELO_K     = 32.0      # per-game K-factor


# ═══════════════════════════════════════════════════════════════════════════
# Model discovery
# ═══════════════════════════════════════════════════════════════════════════

class ModelSpec:
    """Metadata for one trained model."""

    def __init__(self, name: str, model_dir: Path, weights_path: Path) -> None:
        self.name         = name
        self.model_dir    = model_dir
        self.weights_path = weights_path

    def __repr__(self) -> str:
        return f"ModelSpec({self.name})"


def _find_weights(model_dir: Path) -> Optional[Path]:
    """Return the best available weights file in a model directory."""
    # Preference order
    candidates = [
        model_dir / "checkpoints" / "model_final.pt",
        # fall back to the highest-numbered snapshot
        *sorted(
            (model_dir / "checkpoints").glob("model_iter_*.pt"),
            reverse=True,
        ),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def discover_models(names: Optional[list[str]] = None) -> list[ModelSpec]:
    """
    Auto-discover models from  project/models/*/

    Parameters
    ----------
    names : If given, only include models whose folder name is in this list.
    """
    models_root = ROOT / "models"
    if not models_root.exists():
        print(f"  [warn] models/ directory not found at {models_root}")
        return []

    specs = []
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if names and model_dir.name not in names:
            continue

        weights = _find_weights(model_dir)
        if weights is None:
            print(f"  [skip] {model_dir.name} — no weights found in checkpoints/")
            continue

        specs.append(ModelSpec(
            name=model_dir.name,
            model_dir=model_dir.resolve(),
            weights_path=weights.resolve(),
        ))
        print(f"  found : {model_dir.name:<20}  ← {weights.relative_to(ROOT)}")

    return specs


# ═══════════════════════════════════════════════════════════════════════════
# Persistent Elo ratings
# ═══════════════════════════════════════════════════════════════════════════

def load_ratings() -> dict[str, float]:
    """Load ratings from disk.  Returns {} if file does not exist yet."""
    if RATINGS_FILE.exists():
        with open(RATINGS_FILE) as f:
            return json.load(f)
    return {}


def save_ratings(ratings: dict[str, float]) -> None:
    """Write ratings to disk atomically."""
    RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = RATINGS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ratings, f, indent=2, sort_keys=True)
    tmp.replace(RATINGS_FILE)


def ensure_ratings(ratings: dict[str, float], specs: list[ModelSpec]) -> None:
    """Add any new model to the ratings dict at ELO_START."""
    for spec in specs:
        if spec.name not in ratings:
            ratings[spec.name] = ELO_START
            print(f"  new model '{spec.name}' initialised at Elo {ELO_START:.0f}")


def _expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def update_elo(
    ratings: dict[str, float],
    name_a: str,
    name_b: str,
    wins_a: int,
    wins_b: int,
    draws: int,
) -> None:
    """
    Update ratings in-place for a completed match.

    Uses the standard batch formula:
        delta = K × (actual_score − expected_score × total_games)

    which is equivalent to applying the single-game formula after each game
    (assuming ratings stay roughly constant during the match).
    """
    total = wins_a + wins_b + draws
    if total == 0:
        return

    ea      = _expected_score(ratings[name_a], ratings[name_b])
    actual  = wins_a + 0.5 * draws          # score for A (wins + half draws)
    delta   = ELO_K * (actual - total * ea)  # positive if A over-performed

    ratings[name_a] += delta
    ratings[name_b] -= delta


# ═══════════════════════════════════════════════════════════════════════════
# Agent subprocess
# ═══════════════════════════════════════════════════════════════════════════
#
# Why a subprocess per agent?
# ---------------------------
# mcts.py does  `from encoding import encode_state`  at MODULE LEVEL.
# Once imported, the encoding is fixed for that entire process.
# Two models with different encodings CANNOT share a process — the second
# encoding import would be silently ignored (Python caches in sys.modules).
#
# Solution: each model runs in its own subprocess with its own sys.path.
# The coordinator manages the game state and communicates moves via Queues.
# Serialising a 5×5 board (25 ints) between processes is negligible overhead.

def _agent_process(
    model_dir:       str,
    weights_path:    str,
    project_root:    str,
    request_q:       mp.Queue,
    response_q:      mp.Queue,
    num_simulations: int,
    time_limit_s:    Optional[float],
    device_str:      str,
) -> None:
    """
    Long-lived agent process.  Plays multiple games sequentially.

    Message protocol (request_q → agent):
        ('reset',   board, player)               – start a new game
        ('advance', action, board, player)        – opponent moved; advance tree
        ('move',    board, player)                – your turn; put action in response_q
        None                                      – shutdown

    response_q ← agent:
        int   (the chosen action, only in response to 'move')
    """
    import sys
    sys.path.insert(0, project_root)    # engine.py, mcts.py, etc.
    sys.path.insert(0, model_dir)       # this model's config / encoding / model

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from config import CFG
    from model  import build_model
    from mcts   import MCTS
    from engine import ChainReaction, P1, P2

    rows, cols = CFG.rows, CFG.cols
    device     = torch.device(device_str)

    model, _ = build_model(device, compile_model=False)
    state    = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    mcts_obj = MCTS(model, device)

    def _make_game(board: list, player: int) -> ChainReaction:
        """Reconstruct a live ChainReaction from serialised state."""
        g = ChainReaction(rows, cols)
        g.reset()
        g.board          = list(board)
        g.current_player = player
        g.alive[P1]      = sum(1 for x in board if x > 0)
        g.alive[P2]      = sum(1 for x in board if x < 0)
        return g

    current_game: Optional[ChainReaction] = None

    while True:
        msg = request_q.get()

        if msg is None:                     # shutdown signal
            break

        cmd = msg[0]

        if cmd == "reset":
            _, board, player = msg
            mcts_obj.reset()
            current_game = _make_game(board, player)

        elif cmd == "advance":
            _, action, board, player = msg
            new_game = _make_game(board, player)
            mcts_obj.advance_to_action(action, new_game)
            current_game = new_game

        elif cmd == "move":
            _, board, player = msg
            current_game = _make_game(board, player)
            _, action = mcts_obj.run(
                current_game,
                num_simulations=num_simulations,
                temperature=0,
                add_noise=False,
                time_limit_s=time_limit_s,
                reuse_tree=True,
            )
            response_q.put(action)


# ═══════════════════════════════════════════════════════════════════════════
# Starting position generator
# ═══════════════════════════════════════════════════════════════════════════

# Type alias: (board, current_player, opening_move_count)
Position = tuple[list[int], int, int]


def generate_starting_positions(
    num_positions: int,
    max_opening_moves: int = 5,
    rows: int = 5,
    cols: int = 5,
    seed: Optional[int] = None,
) -> list[Position]:
    """
    Generate random opening positions by playing 0–max_opening_moves random
    legal moves from an empty board.

    Each position is drawn independently.  If random play accidentally ends
    the game (extremely unlikely in ≤5 moves on a 5×5 board), that game is
    restarted with one fewer move until a live position is found.

    Returns
    -------
    List of (board, current_player, num_moves_played) tuples.
    """
    rng = _random.Random(seed)
    positions: list[Position] = []

    while len(positions) < num_positions:
        low = 1 if max_opening_moves > 0 else 0
        num_moves = rng.randint(low, max_opening_moves)

        game = ChainReaction(rows, cols)
        game.reset()
        winner = None

        for _ in range(num_moves):
            action = rng.choice(game.legal_actions())
            winner = game.step(action)
            if winner is not None:
                # Game ended (very rare) — discard and try one move fewer
                game = ChainReaction(rows, cols)
                game.reset()
                winner = None
                num_moves = max(0, num_moves - 1)
                break

        positions.append((list(game.board), game.current_player, num_moves))

    return positions


# ═══════════════════════════════════════════════════════════════════════════
# Game coordinator
# ═══════════════════════════════════════════════════════════════════════════

def _play_match(
    spec_a:          ModelSpec,
    spec_b:          ModelSpec,
    positions:       list[Position],
    num_simulations: int,
    time_limit_s:    Optional[float],
    verbose:         bool = True,
) -> tuple[int, int, int, float]:
    """
    Play every position twice — once with spec_a as P1 and once as P2.

    Total games = len(positions) * 2.

    Both agent subprocesses are kept alive for the entire match.
    MCTS trees are reset between games (positions differ each time).

    Returns
    -------
    wins_a, wins_b, draws, avg_game_length
    """
    num_games = len(positions) * 2

    ctx = mp.get_context("spawn")

    req_a: mp.Queue = ctx.Queue()
    res_a: mp.Queue = ctx.Queue()
    req_b: mp.Queue = ctx.Queue()
    res_b: mp.Queue = ctx.Queue()

    proc_a = ctx.Process(
        target=_agent_process,
        args=(str(spec_a.model_dir), str(spec_a.weights_path),
              str(ROOT), req_a, res_a,
              num_simulations, time_limit_s, "cpu"),
        daemon=True,
    )
    proc_b = ctx.Process(
        target=_agent_process,
        args=(str(spec_b.model_dir), str(spec_b.weights_path),
              str(ROOT), req_b, res_b,
              num_simulations, time_limit_s, "cpu"),
        daemon=True,
    )
    proc_a.start()
    proc_b.start()

    wins_a = wins_b = draws = 0
    total_moves = 0
    game_num    = 0

    try:
        for pos_idx, (start_board, start_player, opening_moves) in enumerate(positions):
            # Each position is played exactly twice: once per colour assignment.
            for a_is_p1 in (True, False):
                game_num += 1

                # Map player identities to agent queues for this game.
                if a_is_p1:
                    agents = {P1: (req_a, res_a), P2: (req_b, res_b)}
                else:
                    agents = {P1: (req_b, res_b), P2: (req_a, res_a)}

                # Reconstruct the starting position in the coordinator so
                # we can run the game loop; the agents receive the same
                # board via the 'reset' message.
                game = ChainReaction(5, 5)
                game.reset()
                game.board          = list(start_board)
                game.current_player = start_player
                game.alive[P1]      = sum(1 for x in start_board if x > 0)
                game.alive[P2]      = sum(1 for x in start_board if x < 0)

                req_a.put(("reset", list(start_board), start_player))
                req_b.put(("reset", list(start_board), start_player))

                winner    = None
                moves     = 0
                MAX_MOVES = 70

                while winner is None and moves < MAX_MOVES:
                    current      = game.current_player
                    req_q, res_q = agents[current]

                    req_q.put(("move", list(game.board), game.current_player))
                    action = res_q.get()

                    winner = game.step(action)
                    moves += 1

                    new_board  = list(game.board)
                    new_player = game.current_player
                    req_a.put(("advance", action, new_board, new_player))
                    req_b.put(("advance", action, new_board, new_player))

                total_moves += moves

                if winner is None:
                    draws += 1
                    result_str = "draw"
                elif (winner == P1 and a_is_p1) or (winner == P2 and not a_is_p1):
                    wins_a += 1
                    result_str = spec_a.name
                else:
                    wins_b += 1
                    result_str = spec_b.name

                if verbose:
                    side_a = "P1" if a_is_p1 else "P2"
                    print(f"      pos {pos_idx + 1}/{len(positions)}"
                          f" (opening: {opening_moves}m)"
                          f"  {spec_a.name}={side_a}"
                          f"  →  {result_str}"
                          f"  ({moves} moves)")

    finally:
        req_a.put(None)
        req_b.put(None)
        proc_a.join(timeout=10)
        proc_b.join(timeout=10)
        if proc_a.is_alive():
            proc_a.kill()
        if proc_b.is_alive():
            proc_b.kill()

    avg_len = total_moves / num_games if num_games > 0 else 0.0
    return wins_a, wins_b, draws, avg_len


# ═══════════════════════════════════════════════════════════════════════════
# Leaderboard display
# ═══════════════════════════════════════════════════════════════════════════

def print_leaderboard(
    specs:   list[ModelSpec],
    ratings: dict[str, float],
    session_results: dict[tuple[str, str], dict],
) -> None:
    """Print current standings, sorted by Elo."""
    names   = [s.name for s in specs]
    ranking = sorted(names, key=lambda n: ratings[n], reverse=True)
    col     = max(len(n) for n in names) + 2

    sep = "─" * (col + 46)
    print(f"\n{'═' * (col + 46)}")
    print("  LEADERBOARD")
    print(f"{'═' * (col + 46)}")
    print(f"  {'Model':<{col}}  {'Elo':>6}  {'W':>4}  {'L':>4}  {'D':>4}  {'Win%':>5}")
    print(f"  {sep}")

    for rank, name in enumerate(ranking, 1):
        w = l = d = 0
        for (a, b), res in session_results.items():
            if a == name:
                w += res["wins_a"]; l += res["wins_b"]; d += res["draws"]
            elif b == name:
                w += res["wins_b"]; l += res["wins_a"]; d += res["draws"]
        total = w + l + d
        pct = (w + 0.5 * d) / total if total > 0 else 0.0
        print(f"  {rank}. {name:<{col-3}}  {ratings[name]:>6.0f}"
              f"  {w:>4}  {l:>4}  {d:>4}  {pct:>4.0%}")

    print(f"  {sep}")

    # Pairwise results grid (this session only)
    if len(names) > 1:
        print(f"\n  Pairwise results (this session, row win% vs col)\n")
        header = f"  {'':>{col}}" + "".join(f"  {n[:8]:>8}" for n in ranking)
        print(header)

        for a in ranking:
            row = f"  {a:<{col}}"
            for b in ranking:
                if a == b:
                    row += "      ──"
                    continue
                res = session_results.get((a, b)) or session_results.get((b, a))
                if res is None:
                    row += "     n/a"
                    continue
                if (a, b) in session_results:
                    w, tot = res["wins_a"], res["wins_a"] + res["wins_b"] + res["draws"]
                    d = res["draws"]
                else:
                    w, tot = res["wins_b"], res["wins_a"] + res["wins_b"] + res["draws"]
                    d = res["draws"]
                pct = (w + 0.5 * d) / tot if tot > 0 else 0.0
                row += f"  {pct:>6.0%}"
            print(row)

    print()


# ═══════════════════════════════════════════════════════════════════════════
# History persistence
# ═══════════════════════════════════════════════════════════════════════════

def save_history(
    specs:           list[ModelSpec],
    ratings_before:  dict[str, float],
    ratings_after:   dict[str, float],
    session_results: dict[tuple[str, str], dict],
    args:            argparse.Namespace,
) -> None:
    """Save this tournament run to a timestamped JSON file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = HISTORY_DIR / f"tournament_{ts}.json"

    payload = {
        "timestamp":   datetime.now().isoformat(),
        "settings": {
            "rounds":          args.rounds,
            "positions":       args.positions,
            "max_opening":     args.max_opening,
            "simulations":     args.simulations,
            "time_limit":      args.time_limit,
        },
        "models": [
            {"name": s.name, "weights": str(s.weights_path.relative_to(ROOT))}
            for s in specs
        ],
        "ratings_before": {k: round(v, 1) for k, v in ratings_before.items()},
        "ratings_after":  {k: round(v, 1) for k, v in ratings_after.items()},
        "matches": [
            {
                "model_a":  a,
                "model_b":  b,
                "wins_a":   res["wins_a"],
                "wins_b":   res["wins_b"],
                "draws":    res["draws"],
                "avg_len":  round(res["avg_len"], 1),
                "elo_delta_a": round(
                    ratings_after.get(a, ELO_START) - ratings_before.get(a, ELO_START), 1
                ),
            }
            for (a, b), res in session_results.items()
        ],
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  History saved → {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main tournament loop
# ═══════════════════════════════════════════════════════════════════════════

def run_tournament(args: argparse.Namespace) -> None:
    # ── Discover models ───────────────────────────────────────────────────
    print("\nDiscovering models …")
    specs = discover_models(args.models if args.models else None)

    if len(specs) < 2:
        print(f"\nNeed at least 2 models; found {len(specs)}.  Exiting.")
        sys.exit(1)

    # ── Load / initialise Elo ratings ────────────────────────────────────
    ratings = load_ratings()
    ensure_ratings(ratings, specs)
    ratings_before = {k: v for k, v in ratings.items()}

    time_limit = args.time_limit if args.time_limit > 0 else None

    # Build all unique unordered pairs
    pairs = [
        (specs[i], specs[j])
        for i in range(len(specs))
        for j in range(i + 1, len(specs))
    ]
    total_matches = args.rounds * len(pairs)

    print(f"\nSettings")
    print(f"  Models      : {', '.join(s.name for s in specs)}")
    print(f"  Rounds      : {args.rounds}  ({len(pairs)} pairs × {args.rounds} = {total_matches} matches)")
    print(f"  Positions   : {args.positions}/match  →  {args.positions * 2} games/match (each position played as P1 + P2)")
    print(f"  Opening max : {args.max_opening} random moves")
    print(f"  Simulations : {args.simulations}/move")
    print(f"  Time limit  : {time_limit}s/move")
    print(f"  Elo file    : {RATINGS_FILE.relative_to(ROOT)}\n")

    if args.seed is not None:
        base_seed = args.seed
        print(f"  Position seed : {base_seed}  (fixed — use --seed {base_seed} to reproduce)")
    else:
        import time as _time_mod
        base_seed = _time_mod.time_ns() % (2 ** 31)
        print(f"  Position seed : {base_seed}  (time-based — use --seed {base_seed} to reproduce)")


    session_results: dict[tuple[str, str], dict] = {}
    match_num = 0

    for round_idx in range(1, args.rounds + 1):
        print(f"{'━' * 60}")
        print(f"  Round {round_idx}/{args.rounds}")
        print(f"{'━' * 60}")

        for spec_a, spec_b in pairs:
            match_num += 1

            # Fresh random positions for every matchup.
            # Seed is deterministic: same round + same pair → same positions.
            match_seed = base_seed + round_idx * 10_000 + match_num
            positions  = generate_starting_positions(
                num_positions=args.positions,
                max_opening_moves=args.max_opening,
                seed=match_seed,
            )
            num_games  = len(positions) * 2

            print(f"\n  Match {match_num}/{total_matches}:  "
                  f"{spec_a.name}  vs  {spec_b.name}  "
                  f"({len(positions)} positions × 2 sides = {num_games} games)\n")

            t0 = time.time()
            wins_a, wins_b, draws, avg_len = _play_match(
                spec_a, spec_b,
                positions=positions,
                num_simulations=args.simulations,
                time_limit_s=time_limit,
                verbose=True,
            )
            elapsed = time.time() - t0

            total   = wins_a + wins_b + draws
            wr_a    = (wins_a + 0.5 * draws) / total if total > 0 else 0.0

            print(f"\n    Result: {spec_a.name} {wins_a}W – {wins_b}W {spec_b.name}  "
                  f"draws={draws}  wr={wr_a:.0%}  avg_len={avg_len:.1f}  "
                  f"({elapsed:.0f}s)")

            # Accumulate session totals for this pair
            key = (spec_a.name, spec_b.name)
            if key not in session_results:
                session_results[key] = {"wins_a": 0, "wins_b": 0, "draws": 0, "avg_len": 0.0}
            session_results[key]["wins_a"] += wins_a
            session_results[key]["wins_b"] += wins_b
            session_results[key]["draws"]  += draws
            session_results[key]["avg_len"] = avg_len   # last round's avg

            # Update Elo immediately after each match
            prev_a = ratings[spec_a.name]
            prev_b = ratings[spec_b.name]
            update_elo(ratings, spec_a.name, spec_b.name, wins_a, wins_b, draws)
            save_ratings(ratings)

            print(f"    Elo: {spec_a.name} {prev_a:.0f} → {ratings[spec_a.name]:.0f}  |  "
                  f"{spec_b.name} {prev_b:.0f} → {ratings[spec_b.name]:.0f}")

        # Print leaderboard after each round
        print_leaderboard(specs, ratings, session_results)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"{'═' * 60}")
    print("  FINAL RATINGS")
    print(f"{'═' * 60}")
    for name in sorted(ratings, key=ratings.get, reverse=True):
        before = ratings_before.get(name, ELO_START)
        after  = ratings[name]
        arrow  = f"+{after - before:.0f}" if after >= before else f"{after - before:.0f}"
        print(f"    {name:<20}  {after:>6.0f}  ({arrow})")

    save_history(specs, ratings_before, ratings, session_results, args)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Round-robin tournament with persistent Elo ratings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--models", nargs="+", metavar="NAME",
        help="Folder names under models/ to include (e.g. model_a model_b). "
             "Default: all discovered models.",
    )
    p.add_argument(
        "--rounds", type=int, default=1,
        help="Number of complete round-robins to run.  "
             "Each round plays every pair once.  Default: 1.",
    )
    p.add_argument(
        "--positions", type=int, default=5,
        help="Random starting positions per match.  Each position is played "
             "twice (models swap colours), so total games = positions × 2.  "
             "Default: 5.",
    )
    p.add_argument(
        "--max-opening", type=int, default=5, dest="max_opening",
        help="Maximum number of random moves used to build each starting "
             "position (0 = empty board always, 5 = 0–5 random moves).  "
             "Default: 5.",
    )
    p.add_argument(
        "--seed", type=int, default= None,
        help="Base seed for position generation.  Same seed + round + match "
             "always produces the same positions.  Default: None.",
    )
    p.add_argument(
        "--simulations", type=int, default=16,
        help="MCTS simulations per move for every model.  Default: 16.",
    )
    p.add_argument(
        "--time-limit", type=float, default=0.2, dest="time_limit",
        help="Wall-clock seconds per move (equal for all models). "
             "Pass 0 to disable.  Default: 0.2.",
    )

    args = p.parse_args()
    run_tournament(args)


if __name__ == "__main__":
    main()
