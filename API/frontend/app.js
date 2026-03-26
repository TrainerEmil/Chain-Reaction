/**
 * app.js – Chain Reaction frontend logic
 *
 * Kommuniziert mit dem FastAPI-Backend via fetch().
 * Kein Framework, kein Build-Step – reines vanilla JS.
 */

"use strict";

// ── Config ──────────────────────────────────────────────────────
// Wenn Frontend + Backend auf demselben Server laufen, reicht "/".
// Für separate Entwicklungsserver hier die Backend-URL eintragen.
const API_BASE = "";   // z.B. "http://localhost:8000"

// ── State ───────────────────────────────────────────────────────
let state = null;       // letzter GameState vom Server
let busy  = false;      // sperrt Klicks während KI denkt
let humanPlayer = 1;    // +1 oder -1

// ── DOM refs ─────────────────────────────────────────────────────
const boardEl      = document.getElementById("board");
const statusBar    = document.getElementById("status-bar");
const statusText   = document.getElementById("status-text");
const thinkingOvl  = document.getElementById("thinking-overlay");
const winOverlay   = document.getElementById("win-overlay");
const winEmoji     = document.getElementById("win-emoji");
const winTitle     = document.getElementById("win-title");
const winSub       = document.getElementById("win-sub");
const cardP1       = document.getElementById("card-p1");
const cardP2       = document.getElementById("card-p2");
const arrowP1      = document.getElementById("arrow-p1");
const arrowP2      = document.getElementById("arrow-p2");
const moveLog      = document.getElementById("move-log");
const statMoves    = document.getElementById("stat-moves");
const statActive   = document.getElementById("stat-active");
const modelBadge   = document.getElementById("model-badge");
const simsBadge    = document.getElementById("sims-badge");

// ── API helpers ──────────────────────────────────────────────────
async function api(method, path, body = undefined) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(API_BASE + path, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

// ── Status abrufen ───────────────────────────────────────────────
async function fetchStatus() {
  try {
    const s = await api("GET", "/status");
    // Modellname kürzen
    let src = s.model_source;
    if (src.length > 40) src = "…" + src.slice(-38);
    modelBadge.textContent = src.toUpperCase();
    simsBadge.textContent  = `${s.mcts_sims} SIMS`;
    statActive.textContent = s.active_games;
  } catch (e) {
    modelBadge.textContent = "OFFLINE";
    setStatus("Cannot reach backend — is the server running?", "err");
  }
}

// ── Neues Spiel ──────────────────────────────────────────────────
async function newGame(human = 1) {
  humanPlayer = human;
  busy = false;
  winOverlay.classList.add("hidden");
  moveLog.innerHTML = "";

  try {
    const res = await api("POST", "/new-game", { human_player: human });
    applyState(res.state);
    setStatus(res.message);
    addLog(`Game started — You are P${human === 1 ? "1 (orange)" : "2 (teal)"}`, "");
  } catch (e) {
    setStatus("Error: " + e.message, "err");
  }

  fetchStatus();
}

// ── Zug ausführen ─────────────────────────────────────────────────
async function makeMove(action) {
  if (busy || !state || state.status !== "ongoing") return;
  if (state.current_player !== humanPlayer) return;

  const sessionId = state.session_id;
  busy = true;
  setThinking(true);

  try {
    const res = await api("POST", "/move", { session_id: sessionId, action });

    if (!res.valid) {
      setStatus(res.message, "err");
      busy = false;
      setThinking(false);
      return;
    }

    // Flash der angeklickten Zelle
    const prevBoard = state.board.slice();
    applyState(res.state);

    // Log: menschlicher Zug
    const hr = Math.floor(action / state.cols);
    const hc = action % state.cols;
    addLog(`You: (${hr},${hc})`, "log-p1");

    // Log: KI-Zug
    if (res.ai_action !== null && res.ai_action !== undefined) {
      const ar = Math.floor(res.ai_action / res.state.cols);
      const ac = res.ai_action % res.state.cols;
      addLog(`AI:  (${ar},${ac})`, "log-p2");
    }

    // Statusmeldung
    if (res.state.status === "ongoing") {
      setStatus("Your turn");
    } else {
      handleGameOver(res.state, res.message);
    }

  } catch (e) {
    setStatus("Error: " + e.message, "err");
  } finally {
    busy = false;
    setThinking(false);
    fetchStatus();
  }
}

// ── Spielzustand rendern ──────────────────────────────────────────
function applyState(s) {
  state = s;
  renderBoard();
  renderSidebar();
  statMoves.textContent = s.move_count;
}

function renderBoard() {
  if (!state) return;

  const { rows, cols, board, cap, current_player, status, last_move } = state;

  boardEl.style.gridTemplateColumns = `repeat(${cols}, var(--cell-size))`;

  // Rebuild cells only when dimensions change
  if (boardEl.children.length !== rows * cols) {
    boardEl.innerHTML = "";
    for (let i = 0; i < rows * cols; i++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.idx = i;
      cell.addEventListener("click", () => onCellClick(i));
      boardEl.appendChild(cell);
    }
  }

  const cells = boardEl.querySelectorAll(".cell");
  const gameOver = status !== "ongoing";

  cells.forEach((cell, i) => {
    const v   = board[i];
    const c   = cap[i];
    const abs = Math.abs(v);
    const own = v !== 0 && v * current_player > 0;
    const opp = v !== 0 && v * current_player < 0;

    // Legal?
    const legal = !gameOver && (v === 0 || own);

    cell.className = "cell";
    if (v > 0) cell.classList.add("p1");
    if (v < 0) cell.classList.add("p2");
    if (legal)  cell.classList.add("legal");
    if (!legal && !gameOver) cell.classList.add("illegal");
    if (i === last_move) cell.classList.add("ai-last");

    const isCrit = abs > 0 && abs === c - 1;
    if (isCrit && v > 0) cell.classList.add("critical-p1");
    if (isCrit && v < 0) cell.classList.add("critical-p2");

    // Orbs
    cell.innerHTML = "";
    if (abs > 0) {
      const dotsDiv = document.createElement("div");
      dotsDiv.className = "orb-dots";
      const cls = v > 0 ? "p1" : "p2";
      for (let k = 0; k < Math.min(abs, 4); k++) {
        const orb = document.createElement("div");
        orb.className = `orb ${cls}`;
        orb.style.animationDelay = `${k * 40}ms`;
        dotsDiv.appendChild(orb);
      }
      cell.appendChild(dotsDiv);
    }

    // Capacity hint
    const capEl = document.createElement("div");
    capEl.className = "cell-cap";
    capEl.textContent = c;
    cell.appendChild(capEl);
  });
}

function renderSidebar() {
  if (!state) return;

  const { current_player, status } = state;
  const isHumanTurn = current_player === humanPlayer && status === "ongoing";

  // Player 1 card (human if human_player=1, else AI)
  const p1IsHuman = humanPlayer === 1;

  cardP1.classList.toggle("active", isHumanTurn  === p1IsHuman);
  cardP2.classList.toggle("active", !isHumanTurn  !== p1IsHuman);

  // Simpler: just highlight whose turn it is
  const p1Turn = current_player === 1 && status === "ongoing";
  const p2Turn = current_player === -1 && status === "ongoing";

  cardP1.classList.toggle("active", p1Turn);
  cardP1.classList.toggle("ai-turn", p1IsHuman ? false : p1Turn);
  cardP2.classList.toggle("active", p2Turn);
  cardP2.classList.toggle("ai-turn", p1IsHuman ? p2Turn : false);
}

// ── Game over ────────────────────────────────────────────────────
function handleGameOver(s, message) {
  const won = s.status === "human_won";
  const lost = s.status === "ai_won";

  statusBar.classList.toggle("status-win",  won);
  statusBar.classList.toggle("status-lose", lost);
  statusText.textContent = message;

  winEmoji.textContent = won ? "🎉" : "🤖";
  winTitle.textContent = won ? "YOU WIN" : "AI WINS";
  winTitle.classList.toggle("lose", lost);
  winSub.textContent   = won ? "Well played!" : "Better luck next time.";
  winOverlay.classList.remove("hidden");
}

// ── Zell-Klick ───────────────────────────────────────────────────
function onCellClick(idx) {
  if (busy || !state || state.status !== "ongoing") return;
  if (state.current_player !== humanPlayer) return;
  makeMove(idx);
}

// ── Status helpers ───────────────────────────────────────────────
function setStatus(msg, type = "") {
  statusText.textContent = msg;
  statusBar.classList.remove("status-win", "status-lose", "status-err");
  if (type) statusBar.classList.add("status-" + type);
}

function setThinking(on) {
  thinkingOvl.classList.toggle("hidden", !on);
  if (on) setStatus("AI is thinking…");
}

// ── Move log ──────────────────────────────────────────────────────
function addLog(text, cls) {
  const li = document.createElement("li");
  li.textContent = `#${moveLog.children.length + 1}  ${text}`;
  if (cls) li.classList.add(cls);
  moveLog.prepend(li);   // neueste oben
}

// ── Buttons ───────────────────────────────────────────────────────
document.getElementById("btn-new-p1").addEventListener("click", () => newGame(1));
document.getElementById("btn-new-p2").addEventListener("click", () => newGame(-1));
document.getElementById("btn-play-again").addEventListener("click", () => newGame(humanPlayer));

// ── Boot ──────────────────────────────────────────────────────────
(async () => {
  await fetchStatus();
  await newGame(1);
})();
