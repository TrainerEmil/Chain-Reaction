"""
mcts.py – AlphaZero-style Monte Carlo Tree Search.

Key design decisions:
- A single MCTSNode stores N, W, Q, P for each child action.
- Expansion calls the network once per new leaf (no random rollouts).
- Values are always stored from the perspective of the player who
  *created* the node (the player to move at that node), then negated
  when backpropagating to the parent.
- Dirichlet noise is injected at the root during self-play to ensure
  exploration of non-obvious moves.
- run() accepts an optional time_limit_s: once the budget is spent the
  simulation loop exits early, giving a cheap per-move time cap.
"""

from __future__ import annotations

import math
import time as _time
from typing import Dict, List, Optional

import numpy as np
import torch

from config import CFG
from encoding import encode_state_tensor, legal_action_mask
from engine import ChainReaction


class MCTSNode:
    """
    Represents a single node (= game state) in the search tree.

    Attributes
    ----------
    parent        : Parent node, or None if root.
    action        : The action that led to this node from the parent.
    game          : A *cloned* game state at this node.
    prior         : Prior probability P(a | s) from the network.
    children      : Mapping from action index -> child MCTSNode.
    N             : Visit count.
    W             : Total accumulated value.
    Q             : Mean value  (W / N, or 0 if N == 0).
    is_terminal   : True if the game ended at this node.
    is_expanded   : True once children have been created.
    _leaf_value   : Network value estimate stored during expansion.
    """

    __slots__ = (
        "parent", "action", "game", "prior",
        "children", "N", "W", "is_terminal", "is_expanded",
        "_leaf_value",
    )

    def __init__(
        self,
        game: ChainReaction,
        prior: float = 0.0,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
    ) -> None:
        self.game = game
        self.prior = prior
        self.parent = parent
        self.action = action
        self.children: Dict[int, "MCTSNode"] = {}
        self.N: int = 0
        self.W: float = 0.0
        self.is_terminal: bool = False
        self.is_expanded: bool = False
        self._leaf_value: float = 0.0   # set properly in _expand()

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def puct_score(self, total_visits: int, c_puct: float) -> float:
        """PUCT selection score."""
        u = c_puct * self.prior * math.sqrt(total_visits) / (1 + self.N)
        return -self.Q + u


class MCTS:
    """
    Monte Carlo Tree Search driver.

    Usage
    -----
    mcts = MCTS(model, device)

    # Fixed simulation count (self-play):
    policy, action = mcts.run(game, num_simulations=64, temperature=1.0)

    # Time-limited (evaluation):
    policy, action = mcts.run(game, time_limit_s=0.5, temperature=0)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        c_puct: float = CFG.c_puct,
        dirichlet_alpha: float = CFG.dirichlet_alpha,
        dirichlet_epsilon: float = CFG.dirichlet_epsilon,
    ) -> None:
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        game: ChainReaction,
        num_simulations: int = CFG.mcts_simulations,
        temperature: float = 1.0,
        add_noise: bool = False,
        time_limit_s: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Run MCTS iterations starting from *game*.

        Stopping condition - whichever triggers first:
          1. *num_simulations* iterations completed.
          2. Wall-clock time exceeds *time_limit_s* seconds
             (only checked when time_limit_s is not None).

        At least ONE simulation is always executed so the returned
        policy is never empty even if the budget is already exceeded
        when the loop starts.

        Parameters
        ----------
        game            : Current game state (will NOT be modified).
        num_simulations : Hard upper bound on iterations.
        temperature     : Sampling temperature for action selection.
                          0 or near-zero -> greedy argmax.
                          1.0            -> sample proportional to visit count.
        add_noise       : Inject Dirichlet noise at root (self-play only).
        time_limit_s    : Optional wall-clock budget in seconds per move.
                          When set, the loop exits as soon as the budget
                          is exhausted, regardless of simulation count.
                          Use this during evaluation to bound CPU time.

        Returns
        -------
        policy : np.ndarray of shape (rows * cols,)
        action : int
        """
        root = MCTSNode(game=game.clone())
        self._expand(root)

        if add_noise:
            self._add_dirichlet_noise(root)

        # Compute the absolute deadline once using the monotonic clock.
        # monotonic() is unaffected by system clock changes.
        deadline: Optional[float] = (
            _time.monotonic() + time_limit_s if time_limit_s is not None else None
        )

        for sim in range(num_simulations):
            # Always run at least one simulation (sim == 0) so we always
            # have a non-empty visit distribution to build the policy from.
            # After that, check the clock before every new iteration.
            if sim > 0 and deadline is not None and _time.monotonic() >= deadline:
                break

            node = self._select(root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        return self._build_policy(root, temperature)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode) -> None:
        """
        Expand a leaf: query the network, create child nodes with priors.
        """
        if node.is_expanded or node.is_terminal:
            return

        game = node.game
        state_tensor = encode_state_tensor(game).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits, value_tensor = self.model(state_tensor)

        # Legal mask - set illegal cells to -inf before softmax
        mask = legal_action_mask(game)                   # shape (n,)
        logits_np = logits.squeeze(0).cpu().numpy()
        logits_np[mask == 0.0] = -1e9
        priors = _softmax(logits_np)                     # shape (n,)

        # Store the value at this node (from current player's perspective)
        node._leaf_value = float(value_tensor.item())

        # Create child nodes for all legal actions
        legal = np.where(mask == 1.0)[0]
        for a in legal:
            child_game = game.clone()
            winner = child_game.step(int(a))
            child = MCTSNode(
                game=child_game,
                prior=float(priors[a]),
                parent=node,
                action=int(a),
            )
            if winner is not None:
                child.is_terminal = True
                child._leaf_value = -1.0
            node.children[int(a)] = child

        node.is_expanded = True

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Walk down the tree following PUCT until we reach an unexpanded leaf."""
        node = root
        while node.is_expanded and not node.is_terminal:
            total_n = sum(c.N for c in node.children.values())
            node = max(
                node.children.values(),
                key=lambda c: c.puct_score(total_n, self.c_puct),
            )
        return node

    def _evaluate(self, node: MCTSNode) -> float:
        """Return the value estimate for *node*, expanding it if needed."""
        if node.is_terminal:
            return node._leaf_value
        if not node.is_expanded:
            self._expand(node)
        return node._leaf_value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate *value* from *node* back to the root, negating at each level."""
        current = node
        v = value
        while current is not None:
            current.N += 1
            current.W += v
            v = -v
            current = current.parent

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into the root's child priors."""
        children = list(root.children.values())
        if not children:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        eps = self.dirichlet_epsilon
        for child, n in zip(children, noise):
            child.prior = (1 - eps) * child.prior + eps * n

    def _build_policy(
        self, root: MCTSNode, temperature: float
    ) -> tuple[np.ndarray, int]:
        """Construct the policy target from the root's visit counts."""
        n_actions = root.game.n
        visits = np.zeros(n_actions, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.N

        if temperature == 0 or temperature < 1e-6:
            best = int(np.argmax(visits))
            policy = np.zeros(n_actions, dtype=np.float32)
            policy[best] = 1.0
            return policy, best

        visit_temp = visits ** (1.0 / temperature)
        total = visit_temp.sum()
        if total == 0:
            legal = legal_action_mask(root.game)
            visit_temp = legal
            total = legal.sum()
        policy = visit_temp / total
        action = int(np.random.choice(n_actions, p=policy))
        return policy, action


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()
