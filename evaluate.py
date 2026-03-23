"""
evaluate.py – Pit two agents against each other over multiple games.

Two agent types:
  ModelAgent  – uses time-limited greedy MCTS (no noise).
  RandomAgent – picks a uniformly random legal action.

Evaluation is symmetric: half the games each agent starts as P1.

Time budget
-----------
ModelAgent accepts a time_limit_s parameter (default: CFG.eval_time_limit_s).
When set, MCTS stops adding simulations as soon as the per-move wall-clock
budget is exhausted.  This is the primary knob for controlling evaluation
CPU time.  num_simulations remains as a hard upper cap so the agent never
runs more than that even on a fast machine.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

from config import CFG
from engine import ChainReaction, P1, P2
from mcts import MCTS


# ---------------------------------------------------------------------------
# Agent base class & implementations
# ---------------------------------------------------------------------------

class Agent:
    """Abstract base: any object with a .choose_action(game) method."""

    def choose_action(self, game: ChainReaction) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    """Picks a uniformly random legal action."""

    def choose_action(self, game: ChainReaction) -> int:
        return random.choice(game.legal_actions())


class ModelAgent(Agent):
    """
    Uses MCTS with the given model.

    During evaluation:
      - No Dirichlet noise (deterministic priors).
      - Greedy action selection (temperature = 0).
      - Per-move time budget via time_limit_s to cap CPU use.

    Parameters
    ----------
    model           : The network to query.
    device          : Torch device.
    num_simulations : Hard upper bound on MCTS iterations per move.
    time_limit_s    : Wall-clock budget per move in seconds.
                      None means run all num_simulations (no time cap).
                      Overrides the CFG default when provided explicitly.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations: int = CFG.mcts_simulations,
        time_limit_s: Optional[float] = CFG.eval_time_limit_s,
    ) -> None:
        self.mcts = MCTS(model, device)
        self.num_simulations = num_simulations
        self.time_limit_s = time_limit_s

    def choose_action(self, game: ChainReaction) -> int:
        _, action = self.mcts.run(
            game,
            num_simulations=self.num_simulations,
            temperature=0,              # greedy – no randomness in evaluation
            add_noise=False,            # no exploration noise
            time_limit_s=self.time_limit_s,
        )
        return action


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_one_game(
    agent_p1: Agent,
    agent_p2: Agent,
    rows: int = CFG.rows,
    cols: int = CFG.cols,
    max_length: int = CFG.max_game_length,
) -> Tuple[Optional[int], int]:
    """
    Play one game between *agent_p1* (moves first) and *agent_p2*.

    Returns
    -------
    winner    : P1, P2, or None (draw / timeout).
    num_moves : Total moves played.
    """
    game = ChainReaction(rows, cols)
    game.reset()
    agents = {P1: agent_p1, P2: agent_p2}
    move_count = 0
    winner = None

    while winner is None and move_count < max_length:
        agent = agents[game.current_player]
        action = agent.choose_action(game)
        winner = game.step(action)
        move_count += 1

    return winner, move_count


# ---------------------------------------------------------------------------
# Full evaluation (symmetric)
# ---------------------------------------------------------------------------

def evaluate(
    candidate: Agent,
    opponent: Agent,
    num_games: int = CFG.eval_games,
    rows: int = CFG.rows,
    cols: int = CFG.cols,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Play *num_games* games, alternating who starts.

    Parameters
    ----------
    candidate : The agent we are evaluating (the "new" model).
    opponent  : Baseline (random or old model).
    num_games : Total games; half with candidate as P1, half as P2.

    Returns
    -------
    dict with 'win_rate', 'wins', 'losses', 'draws', 'avg_length'.
    """
    wins = losses = draws = 0
    total_length = 0
    half = num_games // 2

    for i in range(num_games):
        if i < half:
            a1, a2 = candidate, opponent
            cand_player = P1
        else:
            a1, a2 = opponent, candidate
            cand_player = P2

        winner, length = play_one_game(a1, a2, rows, cols)
        total_length += length

        if winner is None:
            draws += 1
        elif winner == cand_player:
            wins += 1
        else:
            losses += 1

    win_rate = wins / num_games
    avg_length = total_length / num_games

    if verbose:
        print(
            f"  Evaluation: W={wins}  L={losses}  D={draws}  "
            f"WinRate={win_rate:.2%}  AvgLen={avg_length:.1f}"
        )

    return {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_length": avg_length,
    }


def evaluate_vs_random(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int = CFG.eval_games,
    time_limit_s: Optional[float] = CFG.eval_time_limit_s,
) -> dict[str, float]:
    """Convenience wrapper: model vs. random agent."""
    lbl = f"{time_limit_s:.2f}s/move" if time_limit_s is not None else "no time cap"
    print(f"  -> Evaluating vs. Random  ({lbl}) ...")
    candidate = ModelAgent(model, device, time_limit_s=time_limit_s)
    opponent  = RandomAgent()
    return evaluate(candidate, opponent, num_games)


def evaluate_vs_model(
    candidate_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    device: torch.device,
    num_games: int = CFG.eval_games,
    time_limit_s: Optional[float] = CFG.eval_time_limit_s,
) -> dict[str, float]:
    """Convenience wrapper: new model vs. old model."""
    lbl = f"{time_limit_s:.2f}s/move" if time_limit_s is not None else "no time cap"
    print(f"  -> Evaluating vs. previous best model  ({lbl}) ...")
    candidate = ModelAgent(candidate_model, device, time_limit_s=time_limit_s)
    opponent  = ModelAgent(opponent_model,  device, time_limit_s=time_limit_s)
    return evaluate(candidate, opponent, num_games)
