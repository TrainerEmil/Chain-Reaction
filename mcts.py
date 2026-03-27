"""
mcts.py – AlphaZero-style Monte Carlo Tree Search.

Key design decisions (unchanged):
- A single MCTSNode stores N, W, Q, P for each child action.
- Expansion calls the network once per new leaf (no random rollouts).
- Values are stored from the perspective of the player who *created* the
  node, and are negated when backpropagating to the parent.
- Dirichlet noise is injected at the root during self-play.
- run() accepts an optional time_limit_s for per-move wall-clock budgets.

    Virtual-loss sign convention for this code
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This MCTS stores values from the CHILD player's perspective.
    The PUCT formula here is:

        score = −Q + U     where Q = W / N

    A child with Q > 0 (tends to win) gets a LOW score from the
    parent's view.  The parent prefers children with LOW Q → high −Q.

    To DISCOURAGE re-selection of an already-selected leaf within the
    same batch, we need to make its score *lower*, i.e. make (−Q) smaller,
    i.e. make Q *larger*.  Therefore virtual loss ADDS to both W and N:

        node.W += _VL,  node.N += _VL   (apply)
        node.W -= _VL,  node.N -= _VL   (remove)

    This is the OPPOSITE sign to standard AlphaGo/Zero implementations
    that use (+Q + U) and therefore subtract from W.

[4] Batch encode via np.stack + single torch.from_numpy
    Avoids B separate unsqueeze() and tensor-creation calls.
"""

from __future__ import annotations

import math
import time as _time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import CFG
from encoding import encode_state, encode_state_tensor, legal_action_mask
from engine import ChainReaction


# ---------------------------------------------------------------------------
# Virtual-loss constant
# ---------------------------------------------------------------------------

# Each selected-but-not-yet-evaluated leaf gets W and N bumped by _VL.
# Value 1 is sufficient to redirect subsequent selections within a batch
# of typical size (4–16).
_VL: int = 1


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    """One node in the MCTS search tree."""

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
        """
        PUCT selection score:  −Q + U
        (see module docstring for the sign convention)
        """
        u = c_puct * self.prior * math.sqrt(max(total_visits, 1)) / (1 + self.N)
        return -self.Q + u


# ---------------------------------------------------------------------------
# MCTS driver
# ---------------------------------------------------------------------------

class MCTS:
    """
    Monte Carlo Tree Search.

    Usage
    -----
    mcts = MCTS(model, device)

    # Self-play with batched inference (fast):
    policy, action = mcts.run(game, num_simulations=12,
                               inference_batch_size=8,
                               temperature=1.0, add_noise=True)

    # Evaluation / legacy single-leaf mode:
    policy, action = mcts.run(game, num_simulations=12,
                               inference_batch_size=1, temperature=0)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        c_puct: float            = CFG.c_puct,
        dirichlet_alpha: float   = CFG.dirichlet_alpha,
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
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.root = None

    def advance_to_action(self, action: int, game: ChainReaction) -> None:
        """Reuse the child subtree after the real game advances."""
        if self.root is None:
            self.root = MCTSNode(game=game.clone())
            return

        child = self.root.children.get(int(action))
        if child is None:
            self.root = MCTSNode(game=game.clone())
            return

        child.parent = None
        self.root = child

    def _same_position(self, a: ChainReaction, b: ChainReaction) -> bool:
        return (
                a.current_player == b.current_player
                and np.array_equal(a.board, b.board)
                and np.array_equal(a.alive, b.alive)
        )

    def run(
        self,
        game: ChainReaction,
        num_simulations: int          = CFG.mcts_simulations,
        temperature: float            = 1.0,
        add_noise: bool               = False,
        time_limit_s: Optional[float] = None,
        reuse_tree: bool              = False,
        inference_batch_size: int     = 1,
    ) -> Tuple[np.ndarray, int]:
        """
        Run MCTS from *game* and return (policy, action).

        Parameters
        ----------
        inference_batch_size : Leaves evaluated per model call.
                               1  → original single-leaf behaviour.
                               >1 → batched mode with virtual loss.
                               Good default: 8 (cuts calls ~6× for 12 sims).
        """
        # ── Set up / reuse root ────────────────────────────────────────
        if reuse_tree and self.root is not None and self._same_position(self.root.game, game):
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

        # ── Simulation loop ────────────────────────────────────────────
        if inference_batch_size > 1:
            self._run_batched(root, num_simulations, inference_batch_size, deadline)
        else:
            for sim in range(num_simulations):
                if sim > 0 and deadline is not None and _time.monotonic() >= deadline:
                    break
                node  = self._select(root)
                value = self._evaluate(node)
                self._backpropagate(node, value)

        return self._build_policy(root, temperature)

    # ------------------------------------------------------------------
    # Batched simulation core
    # ------------------------------------------------------------------

    def _run_batched(
        self,
        root: MCTSNode,
        num_simulations: int,
        batch_size: int,
        deadline: Optional[float],
    ) -> None:
        """
        Run simulations in rounds of up to `batch_size`.

        Each round:
          1. Select up to batch_size leaves using PUCT + virtual loss.
          2. Deduplicate; batch-encode unique unexpanded leaves.
          3. Single model forward pass.
          4. Expand each unique leaf with pre-computed outputs.
          5. Remove virtual loss; backpropagate real values.
        """
        sims_done = 0

        while sims_done < num_simulations:
            if deadline is not None and _time.monotonic() >= deadline:
                break

            n = min(batch_size, num_simulations - sims_done)

            # ── 1. Select n leaves (VL applied per selection) ─────────
            leaves_paths: List[Tuple[MCTSNode, List[MCTSNode]]] = [
                self._select_vl(root) for _ in range(n)
            ]

            # ── 2. Deduplicate leaves needing network expansion ────────
            seen_ids: set = set()
            to_expand: List[MCTSNode] = []
            for leaf, _ in leaves_paths:
                lid = id(leaf)
                if lid not in seen_ids and not leaf.is_terminal and not leaf.is_expanded:
                    seen_ids.add(lid)
                    to_expand.append(leaf)

            # ── 3 & 4. Batch encode → single model call → expand ──────
            if to_expand:
                # One numpy stack + one torch conversion for the whole batch
                encoded_arr = np.stack(
                    [encode_state(node.game) for node in to_expand]
                )                                                    # (B, C, H, W)
                batch_t = torch.from_numpy(
                    np.ascontiguousarray(encoded_arr)
                ).to(self.device, non_blocking=True)

                with torch.inference_mode():
                    logits_batch, values_batch = self.model(batch_t)  # (B,n), (B,1)

                logits_np = logits_batch.cpu().numpy()                # (B, n_actions)
                values_np = values_batch.cpu().numpy()                # (B, 1)

                for i, node in enumerate(to_expand):
                    self._expand_with_outputs(
                        node,
                        logits_np[i],
                        float(values_np[i, 0]),
                    )

            # ── 5. Remove VL; backpropagate ───────────────────────────
            for leaf, path in leaves_paths:
                self._remove_vl(path)
                self._backpropagate(leaf, leaf._leaf_value)

            sims_done += n

    # ------------------------------------------------------------------
    # Virtual-loss helpers
    # ------------------------------------------------------------------

    def _select_vl(
        self, root: MCTSNode
    ) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        Walk the tree with PUCT and return (leaf, path).

        Virtual loss is applied to every node in *path* (from root's
        selected child down to the leaf, inclusive).  This makes later
        selections in the same batch prefer different branches.

        total_n here sums children's N (not just parent.N) because
        virtual-loss increments are not reflected in parent.N until
        backpropagation.  For a 5×5 board k ≤ 25, so O(k) is fast.
        """
        node   = root
        path:  List[MCTSNode] = []
        c_puct = self.c_puct

        while node.is_expanded and not node.is_terminal:
            # Must sum children to account for VL already applied by
            # earlier selections in this batch.
            children = node.children.values()
            total_n  = sum(c.N for c in children) or node.N

            node = max(
                node.children.values(),
                key=lambda child, tn=total_n, cp=c_puct: child.puct_score(tn, cp),
            )
            path.append(node)

        # Apply VL: W += _VL → Q rises → −Q falls → score falls
        for n in path:
            n.N += _VL
            n.W += _VL

        return node, path

    def _remove_vl(self, path: List[MCTSNode]) -> None:
        """Undo virtual loss applied by _select_vl."""
        for n in path:
            n.N -= _VL
            n.W -= _VL

    # ------------------------------------------------------------------
    # Expand with pre-computed outputs
    # ------------------------------------------------------------------

    def _expand_with_outputs(
        self,
        node: MCTSNode,
        logits_np: np.ndarray,
        leaf_value: float,
    ) -> None:
        """
        Expand *node* using network outputs already computed externally.
        Identical logic to _expand() but without the model call.
        """
        if node.is_expanded or node.is_terminal:
            return

        game = node.game
        mask = legal_action_mask(game)

        logits_masked = logits_np.copy()
        logits_masked[mask == 0.0] = -1e9
        priors = _softmax(logits_masked)

        node._leaf_value = leaf_value

        for a in np.nonzero(mask)[0]:
            child_game = game.clone()
            winner     = child_game.step(int(a))
            child      = MCTSNode(
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

    # ------------------------------------------------------------------
    # Single-leaf helpers (root expansion + non-batched path)
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode) -> None:
        """Expand a leaf with a single model call.  Used for root + fallback."""
        if node.is_expanded or node.is_terminal:
            return

        state_tensor = encode_state_tensor(node.game).to(self.device, non_blocking=True)

        with torch.inference_mode():
            logits, value_tensor = self.model(state_tensor)

        logits_np = logits.squeeze(0).detach().cpu().numpy()
        self._expand_with_outputs(node, logits_np, float(value_tensor.item()))

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Walk the tree following PUCT (no virtual loss, O(1) total_n)."""
        node   = root
        c_puct = self.c_puct

        while node.is_expanded and not node.is_terminal:
            total_n = node.N
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
    ) -> Tuple[np.ndarray, int]:
        n_actions = root.game.n
        visits    = np.zeros(n_actions, dtype=np.float32)

        if root.children:
            actions_arr = np.fromiter(root.children.keys(),   dtype=np.int32,
                                      count=len(root.children))
            counts_arr  = np.fromiter(
                (c.N for c in root.children.values()),
                dtype=np.float32, count=len(root.children),
            )
            visits[actions_arr] = counts_arr

        if temperature == 0 or temperature < 1e-6:
            if visits.sum() == 0:
                legal = legal_action_mask(root.game)
                best = int(np.argmax(legal))
            else:
                best = int(np.argmax(visits))
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
    e = np.exp(x - x.max())
    return e / e.sum()
