"""
Chain Reaction – Game Engine
============================================================
Designed for fast self-play and neural-network training.

Board encoding
--------------
  board[i]  is a signed integer:
    0        → empty
   +k        → Player 1 owns cell i with k orbs
   -k        → Player 2 owns cell i with k orbs

Players
-------
  P1 =  1
  P2 = -1
"""

from __future__ import annotations

import copy
from collections import deque
from typing import List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
P1: int = 1
P2: int = -1
EMPTY: int = 0

ActionType = Union[int, Tuple[int, int]]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class OutOfBoundsError(ValueError):
    """Raised when an action refers to a cell outside the board."""


class InvalidMoveError(ValueError):
    """Raised when a player tries to place on an opponent's cell."""


class ReactionLimitError(RuntimeError):
    """Raised when the chain-reaction safety counter is exceeded."""


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------
class ChainReaction:
    """
    Chain Reaction game engine.

    Parameters
    ----------
    rows, cols : int
        Board dimensions (both must be >= 2).
    max_events : int
        Safety limit on the total number of explosions per step.
        Prevents infinite loops on degenerate boards.
    """

    # ------------------------------------------------------------------
    # Construction / initialisation
    # ------------------------------------------------------------------

    def __init__(self, rows: int = 9, cols: int = 6, max_events: int = 200_000) -> None:
        if rows < 2 or cols < 2:
            raise ValueError("Board must be at least 2×2.")
        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.max_events = max_events

        # Pre-compute static topology (neighbours & capacity)
        self.nbrs: List[List[int]] = [[] for _ in range(self.n)]
        self.cap: List[int] = [0] * self.n
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if r > 0:            self.nbrs[i].append((r - 1) * cols + c)
                if r < rows - 1:     self.nbrs[i].append((r + 1) * cols + c)
                if c > 0:            self.nbrs[i].append(r * cols + c - 1)
                if c < cols - 1:     self.nbrs[i].append(r * cols + c + 1)
                self.cap[i] = len(self.nbrs[i])

        # Mutable game state
        self.board: List[int] = [EMPTY] * self.n
        self.current_player: int = P1
        self.alive: dict[int, int] = {P1: 0, P2: 0}

        # Scratch space reused every step (avoids allocation overhead)
        self._queue: deque[int] = deque()
        self._mark: List[int] = [0] * self.n
        self._stamp: int = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def opponent(player: int) -> int:
        """Return the opponent of *player*."""
        return -player          # P1→P2, P2→P1  (equivalent to 3-player for {1,-1})

    def rc_to_index(self, r: int, c: int) -> int:
        """Convert (row, col) to a flat board index."""
        return r * self.cols + c

    def index_to_rc(self, i: int) -> Tuple[int, int]:
        """Convert a flat board index to (row, col)."""
        return divmod(i, self.cols)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the board to the initial (empty) state."""
        for i in range(self.n):
            self.board[i] = EMPTY
        self.current_player = P1
        self.alive[P1] = 0
        self.alive[P2] = 0

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def convert_action(self, action: ActionType) -> int:
        """
        Accept either a flat index *int* or a *(row, col)* tuple and
        return a validated flat index.
        """
        if isinstance(action, (tuple, list)):
            r, c = action
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise OutOfBoundsError(
                    f"(r={r}, c={c}) is outside the {self.rows}×{self.cols} board."
                )
            return r * self.cols + c
        else:
            idx = int(action)
            if not (0 <= idx < self.n):
                raise OutOfBoundsError(
                    f"Index {idx} is outside [0, {self.n - 1}]."
                )
            return idx

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def legal_actions(self) -> List[int]:
        """
        Return a list of all cell indices the current player may place on.
        A cell is legal if it is empty OR already owned by the current player.
        """
        s = self.current_player
        return [i for i in range(self.n) if self.board[i] == EMPTY or self.board[i] * s > 0]

    def is_legal(self, action: ActionType) -> bool:
        """Return True if *action* is a legal move for the current player."""
        try:
            i = self.convert_action(action)
        except OutOfBoundsError:
            return False
        return self.board[i] == EMPTY or self.board[i] * self.current_player > 0

    # ------------------------------------------------------------------
    # Observation / state snapshot
    # ------------------------------------------------------------------

    def observation(self) -> Tuple[Tuple[int, ...], int]:
        """
        Return an immutable snapshot: (board_as_tuple, current_player).
        Suitable as a dict key or for feeding into a neural network.
        """
        return (tuple(self.board), self.current_player)

    def clone(self) -> "ChainReaction":
        """
        Return a deep copy of the engine in its current state.
        Useful for tree search (MCTS, minimax) without polluting the
        original game.
        """
        other = ChainReaction.__new__(ChainReaction)
        other.rows = self.rows
        other.cols = self.cols
        other.n = self.n
        other.max_events = self.max_events
        # Topology is immutable – share references for speed
        other.nbrs = self.nbrs
        other.cap = self.cap
        # Copy mutable state
        other.board = self.board[:]
        other.current_player = self.current_player
        other.alive = self.alive.copy()
        # Fresh scratch space
        other._queue = deque()
        other._mark = [0] * self.n
        other._stamp = 0
        return other

    # ------------------------------------------------------------------
    # Winner check
    # ------------------------------------------------------------------

    def winner(self) -> Optional[int]:
        """
        Return P1, P2, or None (game still in progress).

        A player can only lose *after* both players have had at least
        one move, i.e. after the opponent has more than 2 cells.
        """
        if self.alive[P1] == 0 and self.alive[P2] > 2:
            return P2
        if self.alive[P2] == 0 and self.alive[P1] > 2:
            return P1
        return None

    # ------------------------------------------------------------------
    # Step (core logic)
    # ------------------------------------------------------------------

    def step(self, action: ActionType) -> Optional[int]:
        """
        Place one orb for the current player, resolve all chain reactions,
        then switch the active player.

        Parameters
        ----------
        action : int or (row, col)
            Target cell.

        Returns
        -------
        int or None
            The winning player (P1 or P2) if the game just ended,
            otherwise None.

        Raises
        ------
        OutOfBoundsError    – action is off the board
        InvalidMoveError    – cell belongs to the opponent
        ReactionLimitError  – chain reaction exceeded max_events
        """
        # ---- A: Place one orb ----------------------------------------
        i = self.convert_action(action)
        s = self.current_player
        v = self.board[i]

        if v != EMPTY and v * s < 0:
            raise InvalidMoveError(
                f"Cell {i} belongs to the opponent."
            )

        if v == EMPTY:
            self.alive[s] += 1          # Player gains a new cell
        self.board[i] = v + s           # +1 for P1, -1 for P2

        # ---- B: Seed the reaction queue ------------------------------
        q = self._queue
        q.clear()
        self._stamp += 1
        stamp = self._stamp
        mark = self._mark

        def enqueue(x: int) -> None:
            if mark[x] != stamp:
                mark[x] = stamp
                q.append(x)

        if abs(self.board[i]) >= self.cap[i]:
            enqueue(i)

        # ---- C: Chain reactions (BFS) --------------------------------
        events = 0
        board = self.board
        cap = self.cap
        nbrs = self.nbrs
        alive = self.alive
        opp = self.opponent(s)
        result: Optional[int] = None

        while q:
            x = q.popleft()
            mark[x] = 0
            cx = board[x]
            if cx == EMPTY:
                continue                # Became empty in a previous explosion

            owner_sign = 1 if cx > 0 else -1
            owner = P1 if owner_sign == 1 else P2
            abs_cx = abs(cx)
            cx_cap = cap[x]

            if abs_cx < cx_cap:
                continue                # Stable – nothing to do

            # How many full explosions from this cell?
            t = abs_cx // cx_cap
            events += t
            if events > self.max_events:
                raise ReactionLimitError(
                    f"Chain reaction exceeded {self.max_events} events."
                )

            rem = abs_cx - t * cx_cap
            if rem == 0:
                board[x] = EMPTY
                alive[owner] -= 1
            else:
                board[x] = owner_sign * rem

            # Distribute orbs to neighbours
            for nb in nbrs[x]:
                old = board[nb]
                if old == EMPTY:
                    board[nb] = owner_sign * t
                    alive[owner] += 1
                elif old * owner_sign > 0:          # Same owner
                    board[nb] = old + owner_sign * t
                else:                               # Capture opponent's cell
                    old_owner = P1 if old > 0 else P2
                    alive[old_owner] -= 1
                    alive[owner] += 1
                    board[nb] = owner_sign * (abs(old) + t)

                if abs(board[nb]) >= cap[nb]:
                    enqueue(nb)
                # If the opponent has been eliminated during this chain reaction,
                # end the move immediately instead of continuing to topple a
                # single-colour board indefinitely.
                if alive[opp] == 0 and alive[s] > 2:
                    result = s
                    q.clear()
                    break


        # ---- D: Check winner and switch player -----------------------
        if result is None:
            for j in range(self.n):
                assert self.board[j] == EMPTY or abs(self.board[j]) < self.cap[j], (
                    f"Unstable cell left after resolution at index {j}: "
                    f"value={self.board[j]}, cap={self.cap[j]}"
                )

            result = self.winner()

        if result is None:
            self.current_player = self.opponent(s)

        return result

    # ------------------------------------------------------------------
    # Pretty-printer (useful for debugging)
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = []
        header = "   " + "  ".join(f"{c:2d}" for c in range(self.cols))
        lines.append(header)
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                v = self.board[r * self.cols + c]
                if v == EMPTY:
                    row_cells.append(" . ")
                elif v > 0:
                    row_cells.append(f"+{v} ")
                else:
                    row_cells.append(f"{v} ")
            lines.append(f"{r:2d} " + " ".join(row_cells))
        lines.append(
            f"Turn: {'P1' if self.current_player == P1 else 'P2'} | "
            f"P1 cells: {self.alive[P1]} | P2 cells: {self.alive[P2]}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_game(rows: int = 9, cols: int = 6) -> ChainReaction:
    """Create and reset a fresh ChainReaction game."""
    g = ChainReaction(rows, cols)
    g.reset()
    return g


# ---------------------------------------------------------------------------
# Quick smoke-test (run with:  python chain_reaction.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    random.seed(42)
    game = make_game(rows=6, cols=6)
    print("=== Chain Reaction – smoke test ===\n")
    print(game)

    move_count = 0
    winner = None
    while winner is None:
        actions = game.legal_actions()
        action = random.choice(actions)
        winner = game.step(action)
        move_count += 1

    print(f"\n--- After {move_count} random moves ---")
    print(game)
    print(f"\nWinner: {'P1' if winner == P1 else 'P2'}")
