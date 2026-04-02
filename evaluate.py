"""
evaluate.py – Pit two agents against each other over multiple games.

Two agent types:
  ModelAgent  – greedy MCTS, no noise, with tree reuse between moves.
  RandomAgent – uniformly random legal action.

Evaluation is symmetric: half the games each agent starts as P1.
"""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import torch

from config import CFG
from engine import ChainReaction, P1, P2
from mcts import MCTS


# ---------------------------------------------------------------------------
# Module-level worker state (set by _eval_worker_init)
# ---------------------------------------------------------------------------

_cand_model:   Optional[torch.nn.Module] = None
_opp_model:    Optional[torch.nn.Module] = None   # None → RandomAgent
_worker_device: Optional[torch.device]  = None


def _eval_worker_init(
    cand_state_dict:      Optional[dict],
    opp_state_dict_or_none: Optional[dict],
    device_str:           str, model_dir: str) -> None:
    """
    Called once per worker process when the pool starts.

    Loads both agent models into module-level globals.  Worker tasks only
    receive lightweight scalar arguments – no model data over IPC per game.
    """
    import sys
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    global _cand_model, _opp_model, _worker_device

    # ── Thread pinning ────────────────────────────────────────────────
    # Must happen before any torch tensor operation.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # ── Model loading ─────────────────────────────────────────────────
    from model import build_model  # noqa: PLC0415

    _worker_device = torch.device(device_str)

    _cand_model, _ = build_model(_worker_device, compile_model=False)
    _cand_model.load_state_dict(cand_state_dict)
    _cand_model.eval()

    if opp_state_dict_or_none is not None:
        _opp_model, _ = build_model(_worker_device, compile_model=False)
        _opp_model.load_state_dict(opp_state_dict_or_none)
        _opp_model.eval()
    else:
        _opp_model = None   # signals "use RandomAgent"


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------

class Agent:
    """Base class: choose_action + optional tree-advance hook."""

    def choose_action(self, game: ChainReaction) -> int:
        raise NotImplementedError

    def advance(self, action: int, game: ChainReaction) -> None:
        """Called after *action* is applied to the board.  No-op by default."""
        pass


class RandomAgent(Agent):
    """Picks a uniformly random legal action."""

    def choose_action(self, game: ChainReaction) -> int:
        return random.choice(game.legal_actions())


class ModelAgent(Agent):
    """
    Greedy MCTS agent.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations:      int            = CFG.mcts_simulations,
        time_limit_s:         Optional[float] = CFG.eval_time_limit_s,
        inference_batch_size: int            = CFG.eval_inference_batch_size,
    ) -> None:
        self.mcts                 = MCTS(model, device)
        self.num_simulations      = num_simulations
        self.time_limit_s         = time_limit_s
        self.inference_batch_size = inference_batch_size

    def choose_action(self, game: ChainReaction) -> int:
        _, action = self.mcts.run(
            game,
            num_simulations=self.num_simulations,
            temperature=0,               # greedy – no randomness
            add_noise=False,             # no Dirichlet noise in evaluation
            time_limit_s=self.time_limit_s,
            reuse_tree=True,             # [3] keep subtree for next move
            inference_batch_size=self.inference_batch_size,  # [4]
        )
        return action

    def advance(self, action: int, game: ChainReaction) -> None:
        """
        Re-root the cached MCTS tree at *action*.

        Call this after every board move – even the opponent's – so the
        internal tree stays in sync with the actual game state.
        """
        self.mcts.advance_to_action(action, game)


# ---------------------------------------------------------------------------
# Single game (with tree reuse for both agents)
# ---------------------------------------------------------------------------

def play_one_game(
    agent_p1:   Agent,
    agent_p2:   Agent,
    rows:       int = CFG.rows,
    cols:       int = CFG.cols,
    max_length: int = CFG.max_game_length,
) -> Tuple[Optional[int], int]:
    """
    Play one game between *agent_p1* (moves first) and *agent_p2*.

    After each move both agents' `advance()` hooks are called so that
    ModelAgent instances can re-root their MCTS trees for free.

    Returns
    -------
    winner    : P1, P2, or None (draw / timeout).
    num_moves : Total half-moves played.
    """
    game    = ChainReaction(rows, cols)
    game.reset()
    agents: Dict[int, Agent] = {P1: agent_p1, P2: agent_p2}

    move_count = 0
    winner     = None

    while winner is None and move_count < max_length:
        current = game.current_player
        action  = agents[current].choose_action(game)
        winner  = game.step(action)
        move_count += 1

        # Advance both agents' trees to the position just played.
        # For RandomAgent this is a no-op; for ModelAgent it re-roots
        # the MCTS tree at the played action (free subtree reuse).
        agent_p1.advance(action, game)
        agent_p2.advance(action, game)

    return winner, move_count


# ---------------------------------------------------------------------------
# Worker task
# ---------------------------------------------------------------------------

def _eval_game_worker(
    cand_is_p1:           bool,
    num_simulations:      int,
    time_limit_s:         Optional[float],
    inference_batch_size: int,
    seed:                 Optional[int],
) -> Tuple[Optional[int], int, int]:
    """
    Play one evaluation game inside a worker process.

    Uses module-level _cand_model and _opp_model set by _eval_worker_init.

    Returns (winner, num_moves, cand_player_side) where cand_player_side
    is P1 or P2 so the caller can determine win/loss without re-sending
    the assignment.
    """
    if seed is not None:
        random.seed(seed)

    cand_agent: Agent = ModelAgent(
        _cand_model,   # type: ignore[arg-type]
        _worker_device,  # type: ignore[arg-type]
        num_simulations=num_simulations,
        time_limit_s=time_limit_s,
        inference_batch_size=inference_batch_size,
    )

    if _opp_model is not None:
        opp_agent: Agent = ModelAgent(
            _opp_model,
            _worker_device,  # type: ignore[arg-type]
            num_simulations=num_simulations,
            time_limit_s=time_limit_s,
            inference_batch_size=inference_batch_size,
        )
    else:
        opp_agent = RandomAgent()

    if cand_is_p1:
        a1, a2     = cand_agent, opp_agent
        cand_side  = P1
    else:
        a1, a2     = opp_agent, cand_agent
        cand_side  = P2

    winner, num_moves = play_one_game(a1, a2)
    return winner, num_moves, cand_side


# ---------------------------------------------------------------------------
# Early-stopping helper
# ---------------------------------------------------------------------------

def _check_early_stop(
    wins:       int,
    games_done: int,
    total:      int,
    threshold:  float,
    margin:     int,
    min_games:  int,
) -> Optional[bool]:
    """
    Return True  if we can already accept the candidate (win rate is
                 certain to end above threshold).
    Return False if we can already reject the candidate (win rate is
                 certain to end below threshold).
    Return None  if the outcome is still uncertain.

    Logic (exact, no probability):
      required = threshold * total   (wins needed to cross threshold)

      Accept early: wins are already so high that even winning zero of
        the remaining games still leaves win_rate >= threshold.
        Condition:  wins - margin >= required

      Reject early: wins are so low that even winning ALL remaining
        games cannot reach the threshold.
        Condition:  wins + (total - games_done) + margin < required
    """
    if games_done < min_games:
        return None

    required = threshold * total   # may be fractional

    # Accept: already safely above
    if wins - margin >= required:
        return True

    # Reject: mathematically impossible to reach threshold
    max_possible = wins + (total - games_done)
    if max_possible + margin < required:
        return False

    return None


# ---------------------------------------------------------------------------
# Core parallel evaluation
# ---------------------------------------------------------------------------

def evaluate(
    cand_state_dict:        Optional[dict],
    opp_state_dict_or_none: Optional[dict],
    device:                 torch.device,
    num_games:              int            = CFG.eval_games,
    num_simulations:        int            = CFG.mcts_simulations,
    time_limit_s:           Optional[float] = CFG.eval_time_limit_s,
    inference_batch_size:   int            = CFG.eval_inference_batch_size,
    verbose:                bool           = True,
    seed:                   Optional[int]  = None, model_dir: str = "") -> dict:
    """
    Run *num_games* evaluation games in parallel and return statistics.

    Half the games have the candidate as P1, half as P2 (symmetric).

    Parameters
    ----------
    cand_state_dict        : Candidate model weights.
    opp_state_dict_or_none : Opponent model weights, or None for random.
    device                 : Torch device (workers always use CPU).
    num_games              : Total games to play (before early stopping).
    ...

    Returns
    -------
    dict with 'win_rate', 'wins', 'losses', 'draws', 'avg_length',
    'games_played' (may be < num_games if early stopping fired).
    """
    num_workers = min(os.cpu_count() or 1, num_games)
    half        = num_games // 2

    # Build task list: first half candidate=P1, second half candidate=P2.
    tasks = [(i < half, None if seed is None else seed + i)
             for i in range(num_games)]

    wins = losses = draws = 0
    total_length  = 0
    games_played  = 0
    early_stop_result: Optional[bool] = None

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_eval_worker_init,
        initargs=(cand_state_dict, opp_state_dict_or_none, "cpu", model_dir),
    ) as executor:

        futures = {
            executor.submit(
                _eval_game_worker,
                cand_is_p1,
                num_simulations,
                time_limit_s,
                inference_batch_size,
                game_seed,
            ): idx
            for idx, (cand_is_p1, game_seed) in enumerate(tasks)
        }

        for future in as_completed(futures):
            winner, length, cand_side = future.result()
            total_length += length
            games_played += 1

            if winner is None:
                draws += 1
            elif winner == cand_side:
                wins += 1
            else:
                losses += 1

            # ── Early stopping check ──────────────────────────────────
            if CFG.eval_early_stop:
                decision = _check_early_stop(
                    wins=wins,
                    games_done=games_played,
                    total=num_games,
                    threshold=CFG.win_rate_threshold,
                    margin=CFG.eval_early_stop_margin,
                    min_games=CFG.eval_min_games,
                )
                if decision is not None:
                    early_stop_result = decision
                    # Cancel pending futures – we have enough information.
                    for f in futures:
                        f.cancel()
                    break

    win_rate   = wins / games_played if games_played > 0 else 0.0
    avg_length = total_length / games_played if games_played > 0 else 0.0

    stop_note = ""
    if early_stop_result is not None:
        verdict   = "ACCEPT" if early_stop_result else "REJECT"
        stop_note = f"  [early stop → {verdict} after {games_played}/{num_games} games]"

    if verbose:
        print(
            f"  Evaluation: W={wins}  L={losses}  D={draws}  "
            f"WinRate={win_rate:.2%}  AvgLen={avg_length:.1f}"
            f"  Games={games_played}/{num_games}{stop_note}"
        )

    return {
        "win_rate":     win_rate,
        "wins":         wins,
        "losses":       losses,
        "draws":        draws,
        "avg_length":   avg_length,
        "games_played": games_played,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers (same API as original)
# ---------------------------------------------------------------------------

def _serialize(model: torch.nn.Module) -> dict:
    """Return a CPU copy of the model state dict."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def evaluate_vs_random(
    model:        torch.nn.Module,
    device:       torch.device,
    num_games:    int            = CFG.eval_games,
    time_limit_s: Optional[float] = CFG.eval_time_limit_s,
    model_dir:    str             = "",
) -> dict:
    """Convenience wrapper: model vs. random agent (parallel)."""
    lbl = f"{time_limit_s:.2f}s/move" if time_limit_s is not None else "no time cap"
    print(f"  -> Evaluating vs. Random  ({lbl}) ...")
    return evaluate(
        cand_state_dict=_serialize(model),
        opp_state_dict_or_none=None,
        device=device,
        num_games=num_games,
        time_limit_s=time_limit_s,
        model_dir=model_dir,
    )


def evaluate_vs_model(
    candidate_model: torch.nn.Module,
    opponent_model:  torch.nn.Module,
    device:          torch.device,
    num_games:       int            = CFG.eval_games,
    time_limit_s:    Optional[float] = CFG.eval_time_limit_s,
    model_dir:    str             = "",
) -> dict:
    """Convenience wrapper: new model vs. old model (parallel)."""
    lbl = f"{time_limit_s:.2f}s/move" if time_limit_s is not None else "no time cap"
    print(f"  -> Evaluating vs. previous best model  ({lbl}) ...")
    return evaluate(
        cand_state_dict=_serialize(candidate_model),
        opp_state_dict_or_none=_serialize(opponent_model),
        device=device,
        num_games=num_games,
        time_limit_s=time_limit_s,
        model_dir=model_dir,
    )
