"""
schemas.py – Pydantic-Modelle für Request / Response.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Serialisiertes Spielfeld
# ---------------------------------------------------------------------------

class GameState(BaseModel):
    """Vollständiger, serialisierbarer Spielzustand."""

    session_id:     str
    board:          List[int]       # flat, signed  (+k = P1, -k = P2, 0 = leer)
    cap:            List[int]       # Kapazitäten der Zellen (static)
    rows:           int
    cols:           int
    current_player: int             # +1 (P1/Mensch) oder -1 (P2/KI)
    winner:         Optional[int]   # +1 | -1 | None
    status:         str             # "ongoing" | "human_won" | "ai_won"
    last_move:      Optional[int]   # letzter Zug (flat index)
    move_count:     int


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class NewGameRequest(BaseModel):
    human_player: int = 1   # +1 → Mensch spielt als P1 (zieht zuerst)
                             # -1 → Mensch spielt als P2 (KI zieht zuerst)


class MoveRequest(BaseModel):
    session_id: str
    action:     int     # flat board index


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class NewGameResponse(BaseModel):
    state:    GameState
    message:  str


class MoveResponse(BaseModel):
    state:      GameState
    valid:      bool
    message:    str
    ai_action:  Optional[int] = None   # flat index des KI-Zugs


class StatusResponse(BaseModel):
    model_source: str
    board_size:   str
    mcts_sims:    int
    active_games: int
