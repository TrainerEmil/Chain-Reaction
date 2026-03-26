"""
game_service.py – Verwaltet laufende Spielsitzungen und KI-Züge.

Jede Sitzung wird unter einer UUID gespeichert.  Der Spielzustand
(ChainReaction-Engine + MCTS-Instanz) lebt im Arbeitsspeicher.
Für ein MVP ohne Persistenz ist das ausreichend.

Thread-Sicherheit: FastAPI läuft per Default single-threaded (uvicorn
mit einem Worker). Für Produktion mit mehreren Workern würde man Redis
oder einen ähnlichen externen Store benötigen.
"""

from __future__ import annotations

import uuid
from typing import Dict, Optional, Tuple

import torch

from config import CFG
from engine import ChainReaction, P1, P2
from mcts import MCTS
from schemas import GameState


# ---------------------------------------------------------------------------
# Interne Session-Struktur
# ---------------------------------------------------------------------------

class _Session:
    """Eine laufende Spielpartie."""

    def __init__(
        self,
        game: ChainReaction,
        mcts: MCTS,
        human_player: int,
    ) -> None:
        self.game         = game
        self.mcts         = mcts
        self.human_player = human_player   # P1 (+1) oder P2 (-1)
        self.move_count   = 0
        self.last_move: Optional[int] = None


# ---------------------------------------------------------------------------
# GameService
# ---------------------------------------------------------------------------

class GameService:
    """
    Zentrale Verwaltung aller laufenden Spiele.

    Instanz wird einmal in app.py erzeugt und über den FastAPI-Lifespan
    geteilt.
    """

    def __init__(
        self,
        model:        torch.nn.Module,
        device:       torch.device,
        mcts_sims:    int   = CFG.mcts_simulations,
        mcts_batch:   int   = CFG.eval_inference_batch_size,
        time_limit_s: float = CFG.eval_time_limit_s or 1.0,
    ) -> None:
        self._model       = model
        self._device      = device
        self._mcts_sims   = mcts_sims
        self._mcts_batch  = mcts_batch
        self._time_limit  = time_limit_s
        self._sessions: Dict[str, _Session] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def active_game_count(self) -> int:
        return len(self._sessions)

    def new_game(self, human_player: int = P1) -> Tuple[str, GameState]:
        """
        Erzeugt eine neue Sitzung.

        Falls human_player == P2, macht die KI direkt ihren ersten Zug.
        """
        if human_player not in (P1, P2):
            human_player = P1

        session_id = str(uuid.uuid4())

        game = ChainReaction(CFG.rows, CFG.cols)
        game.reset()

        mcts = MCTS(self._model, self._device)
        session = _Session(game, mcts, human_player)
        self._sessions[session_id] = session

        # Wenn die KI zuerst zieht
        if game.current_player != human_player:
            self._ai_move(session)

        return session_id, self._serialize(session_id, session)

    def human_move(
        self, session_id: str, action: int
    ) -> Tuple[bool, str, Optional[int], GameState]:
        """
        Führt einen menschlichen Zug aus und lässt die KI antworten.

        Returns
        -------
        valid      : True wenn der Zug legal war
        message    : Statusnachricht
        ai_action  : flat index des KI-Zugs (None wenn Spiel vorbei)
        state      : aktualisierter Spielzustand
        """
        session = self._sessions.get(session_id)
        if session is None:
            dummy_state = GameState(
                session_id=session_id, board=[], cap=[], rows=0, cols=0,
                current_player=0, winner=None, status="error",
                last_move=None, move_count=0,
            )
            return False, "Session not found", None, dummy_state

        game = session.game

        # Spielzustand prüfen
        if game.winner() is not None:
            return False, "Game already over", None, self._serialize(session_id, session)

        if game.current_player != session.human_player:
            return False, "Not your turn", None, self._serialize(session_id, session)

        # Legalitätsprüfung
        if not game.is_legal(action):
            return False, f"Illegal move: cell {action}", None, self._serialize(session_id, session)

        # Menschenzug
        session.mcts.advance_to_action(action, game)
        winner = game.step(action)
        session.last_move = action
        session.move_count += 1

        if winner is not None:
            # Spiel vorbei nach menschlichem Zug
            return True, self._end_message(winner, session.human_player), None, self._serialize(session_id, session)

        # KI-Zug
        ai_action = self._ai_move(session)

        winner = game.winner()
        msg = "OK" if winner is None else self._end_message(winner, session.human_player)

        return True, msg, ai_action, self._serialize(session_id, session)

    def get_state(self, session_id: str) -> Optional[GameState]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return self._serialize(session_id, session)

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ai_move(self, session: _Session) -> int:
        """Lässt die KI einen Zug machen. Gibt flat index zurück."""
        game = session.game
        mcts = session.mcts

        _, action = mcts.run(
            game,
            num_simulations=self._mcts_sims,
            temperature=0.0,          # greedy während Evaluation
            add_noise=False,
            reuse_tree=True,
            time_limit_s=self._time_limit,
            inference_batch_size=self._mcts_batch,
        )

        mcts.advance_to_action(action, game)
        game.step(action)
        session.last_move = action
        session.move_count += 1
        return action

    def _serialize(self, session_id: str, session: _Session) -> GameState:
        game   = session.game
        winner = game.winner()

        if winner is None:
            status = "ongoing"
        elif winner == session.human_player:
            status = "human_won"
        else:
            status = "ai_won"

        return GameState(
            session_id     = session_id,
            board          = list(game.board),
            cap            = list(game.cap),
            rows           = game.rows,
            cols           = game.cols,
            current_player = game.current_player,
            winner         = winner,
            status         = status,
            last_move      = session.last_move,
            move_count     = session.move_count,
        )

    @staticmethod
    def _end_message(winner: int, human_player: int) -> str:
        if winner == human_player:
            return "You win! 🎉"
        return "AI wins! Better luck next time."
