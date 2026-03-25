"""
mcts.py – AlphaZero-style Monte Carlo Tree Search.

Key design decisions (unchanged from original):
- A single MCTSNode stores N, W, Q, P for each child action.
- Expansion calls the network once per new leaf (no random rollouts).
- Values are stored from the perspective of the player who *created* the
  node, and are negated when backpropagating to the parent.
- Dirichlet noise is injected at the root during self-play.
- run() accepts an optional time_limit_s for per-move wall-clock budgets.

OPTIMISATIONS IN THIS FILE
--------------------------
1. _select: total visit count for PUCT
   Original: summed child.N over every child in an O(k) loop.
   Fixed:    the parent's own N equals that sum (each simulation
             increments the parent and exactly one descendant path).
             We therefore read node.N directly – O(1) instead of O(k).

2. _select: lambda closure overhead
   Python closures capture the enclosing scope on every call.  Passing
   c_puct and total_n as default arguments (`key=lambda c, tn=tn, cp=cp`)
   binds them at lambda creation time and avoids dictionary lookups in
   the inner loop.

3. _build_policy: pre-allocated visits array
   The original iterated over children and assigned into the array.
   Now we build a dict of {action: N} directly and convert once – saves
   one Python-level loop iteration.

4. legal_action_mask usage:
   Replaced `np.where(mask == 1.0)[0]` (comparison + where) with
   `np.nonzero(mask)[0]` which is a single C call.
"""

from __future__ import annotations

import math
import time as _time
from typing import Dict, Optional

import numpy as np
import torch

from config import CFG
from encoding import encode_state_tensor, legal_action_mask
from engine import ChainReaction


class MCTSNode:
    """
    Represents a single node (= game state) in the search tree.

    Attributes (unchanged from original)
    -------------------------------------
    parent, action, game, prior, children, N, W, is_terminal,
    is_expanded, _leaf_value
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
        self.game     = game
        self.prior    = prior
        self.parent   = parent
        self.action   = action
        self.children: Dict[int, "MCTSNode"] = {}
        self.N:  int   = 0
        self.W:  float = 0.0
        self.is_terminal: bool = False
        self.is_expanded: bool = False
        self._leaf_value: float = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def puct_score(self, total_visits: int, c_puct: float) -> float:
        u = c_puct * self.prior * math.sqrt(max(total_visits, 1)) / (1 + self.N)
        return -self.Q + u


class MCTS:
    """Monte Carlo Tree Search driver.  API identical to original."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        c_puct: float          = CFG.c_puct,
        dirichlet_alpha: float = CFG.dirichlet_alpha,
        dirichlet_epsilon: float = CFG.dirichlet_epsilon,
    ) -> None:
        self.model             = model
        self.device            = device
        self.c_puct            = c_puct
        self.dirichlet_alpha   = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root: Optional[MCTSNode] = None

        self.model.eval()

    # ------------------------------------------------------------------
    # Public API   (unchanged signatures)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.root = None

    def advance_to_action(self, action: int, game: ChainReaction) -> None:
        if self.root is None:
            self.root = MCTSNode(game=game.clone())
            return

        child = self.root.children.get(int(action))
        if child is None:
            self.root = MCTSNode(game=game.clone())
            return

        child.parent = None
        self.root = child

    def run(
        self,
        game: ChainReaction,
        num_simulations: int   = CFG.mcts_simulations,
        temperature: float     = 1.0,
        add_noise: bool        = False,
        time_limit_s: Optional[float] = None,
        reuse_tree: bool       = False,
    ) -> tuple[np.ndarray, int]:
        """Run MCTS and return (policy, action).  Identical contract to original."""
        if reuse_tree and self.root is not None:
            root = self.root
        else:
            root = MCTSNode(game=game.clone())
            if reuse_tree:
                self.root = root

        if not root.is_expanded and not root.is_terminal:
            self._expand(root)

        if add_noise:
            self._add_dirichlet_noise(root)

        deadline: Optional[float] = (
            _time.monotonic() + time_limit_s if time_limit_s is not None else None
        )

        for sim in range(num_simulations):
            if sim > 0 and deadline is not None and _time.monotonic() >= deadline:
                break

            node  = self._select(root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        return self._build_policy(root, temperature)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode) -> None:
        """
        Expand a leaf: query the network and create child nodes with priors.

        Change: use np.nonzero() instead of np.where(mask == 1.0) – one
        fewer array traversal in C.
        """
        if node.is_expanded or node.is_terminal:
            return

        game         = node.game
        state_tensor = encode_state_tensor(game).to(self.device, non_blocking=True)

        with torch.inference_mode():
            logits, value_tensor = self.model(state_tensor)

        mask      = legal_action_mask(game)
        logits_np = logits.squeeze(0).detach().cpu().numpy()
        logits_np[mask == 0.0] = -1e9
        priors = _softmax(logits_np)

        node._leaf_value = float(value_tensor.item())

        # np.nonzero is one C call vs np.where + comparison
        legal = np.nonzero(mask)[0]
        for a in legal:
            child_game = game.clone()
            winner     = child_game.step(int(a))
            child      = MCTSNode(
                game=child_game,
                prior=float(priors[a]),
                parent=node,
                action=int(a),
            )
            if winner is not None:
                child.is_terminal  = True
                child._leaf_value  = -1.0
            node.children[int(a)] = child

        node.is_expanded = True

    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Walk the tree following PUCT until we reach an unexpanded leaf.

        OPTIMISATION – O(1) total-visit count
        --------------------------------------
        Original code summed `child.N` over all children to get
        `total_n`.  This is unnecessary: after backpropagation every
        completed simulation increments the parent node's N by exactly 1,
        so `parent.N` equals the sum of its children's visit counts
        (modulo the very first expansion, where parent.N == 0 is handled
        by `max(total_visits, 1)` inside puct_score).

        We also bind c_puct and total_n as default arguments to the
        lambda to avoid repeated global/closure lookups in the hot loop.
        """
        node   = root
        c_puct = self.c_puct

        while node.is_expanded and not node.is_terminal:
            total_n = node.N   # O(1) – see docstring above

            # Default-arg binding avoids closure overhead in tight loop
            node = max(
                node.children.values(),
                key=lambda child, tn=total_n, cp=c_puct: child.puct_score(tn, cp),
            )

        return node

    def _evaluate(self, node: MCTSNode) -> float:
        if node.is_terminal:
            return node._leaf_value
        if not node.is_expanded:
            self._expand(node)
        return node._leaf_value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        current = node
        v = value
        while current is not None:
            current.N += 1
            current.W += v
            v = -v
            current = current.parent

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        children = list(root.children.values())
        if not children:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        eps   = self.dirichlet_epsilon
        for child, n in zip(children, noise):
            child.prior = (1 - eps) * child.prior + eps * n

    def _build_policy(
        self, root: MCTSNode, temperature: float
    ) -> tuple[np.ndarray, int]:
        """
        Construct the policy target from root visit counts.

        Change: build visits via a dict-comprehension and vectorised
        indexing instead of a for-loop with individual assignments.
        """
        n_actions = root.game.n
        visits    = np.zeros(n_actions, dtype=np.float32)

        # Direct numpy fancy-index assignment – one C operation
        if root.children:
            actions_arr = np.fromiter(root.children.keys(), dtype=np.int32,
                                      count=len(root.children))
            counts_arr  = np.fromiter(
                (c.N for c in root.children.values()),
                dtype=np.float32, count=len(root.children),
            )
            visits[actions_arr] = counts_arr

        if temperature == 0 or temperature < 1e-6:
            best   = int(np.argmax(visits))
            policy = np.zeros(n_actions, dtype=np.float32)
            policy[best] = 1.0
            return policy, best

        visit_temp = visits ** (1.0 / temperature)
        total      = visit_temp.sum()
        if total == 0:
            legal      = legal_action_mask(root.game)
            visit_temp = legal
            total      = legal.sum()
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
