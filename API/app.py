"""
app.py – FastAPI-Server für Chain Reaction vs. KI.

Starten:
  uvicorn app:app --reload --port 8000

Endpoints:
  GET  /status          → Modellinfos + aktive Spiele
  POST /new-game        → Neue Partie starten
  POST /move            → Menschlichen Zug ausführen + KI antwortet
  GET  /state/{id}      → Aktuellen Zustand abrufen
  DELETE /game/{id}     → Sitzung löschen
  GET  /                → Statische index.html ausliefern
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import CFG
from game_service import GameService
from model_loader import load_best_model
from schemas import (
    GameState,
    MoveRequest,
    MoveResponse,
    NewGameRequest,
    NewGameResponse,
    StatusResponse,
)

# ---------------------------------------------------------------------------
# App-weiter Zustand
# ---------------------------------------------------------------------------

_service: GameService | None = None
_model_source: str = "not loaded"


# ---------------------------------------------------------------------------
# Lifespan: Modell beim Start laden
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    global _service, _model_source

    device = torch.device("cpu")
    model, source = load_best_model(device)
    _model_source = source
    _service = GameService(model, device)

    print(f"[ChainReaction] Model loaded: {source}")
    print(f"[ChainReaction] Board: {CFG.rows}×{CFG.cols}  |  MCTS sims: {CFG.mcts_simulations}")

    yield
    # Cleanup (falls nötig)


# ---------------------------------------------------------------------------
# FastAPI-Instanz
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chain Reaction – Play vs. AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS für lokale Frontend-Entwicklung
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statische Dateien (HTML / CSS / JS) aus dem frontend/-Ordner ausliefern
_FRONTEND_DIR = Path(__file__).parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_service() -> GameService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _service


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_index():
    """Liefert das Frontend aus."""
    index = _FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return FileResponse(str(Path(__file__).parent / "index.html"))


@app.get("/status", response_model=StatusResponse)
async def status():
    """Serverinfos: geladenes Modell, Boardgröße, aktive Spiele."""
    svc = _get_service()
    return StatusResponse(
        model_source = _model_source,
        board_size   = f"{CFG.rows}×{CFG.cols}",
        mcts_sims    = CFG.mcts_simulations,
        active_games = svc.active_game_count,
    )


@app.post("/new-game", response_model=NewGameResponse)
async def new_game(req: NewGameRequest = NewGameRequest()):
    """
    Startet eine neue Partie.

    human_player: 1 → du spielst als P1 und ziehst zuerst
                 -1 → du spielst als P2, KI zieht zuerst
    """
    svc = _get_service()
    session_id, state = svc.new_game(human_player=req.human_player)
    return NewGameResponse(
        state   = state,
        message = f"New game started. You are Player {'1' if req.human_player == 1 else '2'}.",
    )


@app.post("/move", response_model=MoveResponse)
async def make_move(req: MoveRequest):
    """
    Menschlicher Zug + KI-Antwort.

    action: flat board index (row * cols + col)
    """
    svc = _get_service()
    valid, message, ai_action, state = svc.human_move(req.session_id, req.action)

    if not valid and state.rows == 0:
        raise HTTPException(status_code=404, detail=message)

    return MoveResponse(
        state     = state,
        valid     = valid,
        message   = message,
        ai_action = ai_action,
    )


@app.get("/state/{session_id}", response_model=GameState)
async def get_state(session_id: str):
    """Gibt den aktuellen Spielzustand zurück."""
    svc   = _get_service()
    state = svc.get_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@app.delete("/game/{session_id}")
async def delete_game(session_id: str):
    """Löscht eine Sitzung (Aufräumen nach Spielende)."""
    svc = _get_service()
    svc.delete_session(session_id)
    return {"deleted": session_id}
