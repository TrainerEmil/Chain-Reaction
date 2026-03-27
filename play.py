"""
play.py – Pygame UI: Human vs. best trained model.

Controls
--------
  Left-click  : Place an orb on a legal cell (your turn).
  R           : Restart the game.
  ESC / Q     : Quit.

Model loading (tried in order)
-------------------------------
  1. checkpoints/training_state.pt  →  best_model_state  (recommended)
  2. checkpoints/model_final.pt     →  full state_dict
  3. checkpoints/model_iter_NNN.pt  →  highest iteration found

Run
---
  python play.py
  python play.py --sims 32          # MCTS simulations for AI
  python play.py --time 1.0         # seconds per AI move (overrides --sims cap)
  python play.py --human-first      # human plays as P1 (default)
  python play.py --ai-first         # AI plays as P1, human as P2
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
from typing import Optional

import pygame
import torch

# ── Project imports ───────────────────────────────────────────────────────────
from config import CFG
from engine import ChainReaction, P1, P2, EMPTY
from evaluate import ModelAgent
from model import build_model


# =============================================================================
# Visual constants
# =============================================================================

CELL_SIZE   = 90          # pixels per board cell
MARGIN      = 50          # border around the grid
INFO_HEIGHT = 110         # status bar below the grid
ORB_RADIUS  = 18          # base orb circle radius
ORB_GAP     = 10          # gap between multi-orb sub-circles
FPS         = 60

# Colour palette
BG_COLOR        = ( 18,  20,  30)   # near-black background
GRID_COLOR      = ( 45,  50,  70)   # grid lines
CELL_EMPTY      = ( 30,  34,  50)   # empty cell fill
P1_COLOR        = ( 72, 175, 240)   # blue – human
P2_COLOR        = (240,  90,  80)   # red  – AI
P1_DARK         = ( 30,  90, 160)
P2_DARK         = (160,  40,  30)
P1_GLOW         = (150, 215, 255)
P2_GLOW         = (255, 160, 140)
LEGAL_TINT      = ( 60,  80,  50)   # subtle green tint on hover
HOVER_ALPHA     = 80
TEXT_COLOR      = (220, 225, 240)
DIM_COLOR       = (100, 110, 140)
WIN_OVERLAY     = ( 18,  20,  30, 200)

FONT_NAME       = None              # None = pygame default monospace


# =============================================================================
# Utility
# =============================================================================

def load_best_model(device: torch.device) -> Optional[torch.nn.Module]:
    """
    Try to load the best model weights. Returns None if no checkpoint found.
    Search order:
      1. checkpoints/training_state.pt   → key 'best_model_state'
      2. checkpoints/model_final.pt
      3. checkpoints/model_iter_NNN.pt   → highest iteration
    """
    ckpt_dir = CFG.checkpoint_dir

    # 1. Full training state (preferred – always contains the accepted best)
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    if os.path.exists(state_path):
        print(f"Loading best model from {state_path} ...")
        payload = torch.load(state_path, map_location=device, weights_only=False)
        model, _ = build_model(device)
        model.load_state_dict(payload["best_model_state"])
        model.eval()
        print("  OK (best_model_state)")
        return model

    # 2. Final checkpoint
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    if os.path.exists(final_path):
        print(f"Loading model from {final_path} ...")
        model, _ = build_model(device)
        model.load_state_dict(torch.load(final_path, map_location=device, weights_only=True))
        model.eval()
        print("  OK (model_final)")
        return model

    # 3. Highest numbered iteration checkpoint
    iter_files = sorted(glob.glob(os.path.join(ckpt_dir, "model_iter_*.pt")))
    if iter_files:
        path = iter_files[-1]
        print(f"Loading model from {path} ...")
        model, _ = build_model(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        print(f"  OK ({os.path.basename(path)})")
        return model

    return None


def orb_positions(count: int, cx: int, cy: int) -> list[tuple[int, int]]:
    """
    Return pixel centres for *count* sub-circles arranged inside a cell.
    Supports 1–4 orbs.
    """
    g = ORB_GAP + ORB_RADIUS
    if count == 1:
        return [(cx, cy)]
    if count == 2:
        return [(cx - g // 2, cy), (cx + g // 2, cy)]
    if count == 3:
        return [(cx, cy - g // 2),
                (cx - g // 2, cy + g // 2),
                (cx + g // 2, cy + g // 2)]
    # 4+: 2×2 grid
    return [(cx - g // 2, cy - g // 2),
            (cx + g // 2, cy - g // 2),
            (cx - g // 2, cy + g // 2),
            (cx + g // 2, cy + g // 2)]


# =============================================================================
# UI state machine
# =============================================================================

class GameUI:
    """Manages the Pygame window and all rendering/input logic."""

    def __init__(
        self,
        ai_agent: Optional[ModelAgent],
        human_player: int,          # P1 or P2
        num_sims: int,
        time_limit: Optional[float],
    ) -> None:
        self.ai_agent    = ai_agent
        self.human       = human_player
        self.ai_player   = P1 if human_player == P2 else P2
        self.num_sims    = num_sims
        self.time_limit  = time_limit

        self.rows = CFG.rows
        self.cols = CFG.cols

        # Window dimensions
        self.win_w = MARGIN * 2 + self.cols * CELL_SIZE
        self.win_h = MARGIN * 2 + self.rows * CELL_SIZE + INFO_HEIGHT

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Chain Reaction – Human vs AI")

        self.clock  = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont(FONT_NAME, 28, bold=True)
        self.font_md = pygame.font.SysFont(FONT_NAME, 20)
        self.font_sm = pygame.font.SysFont(FONT_NAME, 15)

        # Hover surface (semi-transparent tint)
        self.hover_surf = pygame.Surface((CELL_SIZE - 4, CELL_SIZE - 4), pygame.SRCALPHA)
        self.hover_surf.fill((*LEGAL_TINT, HOVER_ALPHA))

        self._new_game()

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def _new_game(self) -> None:
        self.game = ChainReaction(self.rows, self.cols)
        self.game.reset()
        self.winner: Optional[int] = None
        self.ai_thinking = False
        self.hover_cell: Optional[int] = None
        self.status_msg = ""

        if not hasattr(self, "game_id"):
            self.game_id = 0
        self.game_id += 1

        self._update_status()

        # If AI goes first, kick off its move immediately
        if self.game.current_player == self.ai_player:
            self._trigger_ai_move()

    def _update_status(self) -> None:
        if self.winner is not None:
            who = "You win! 🎉" if self.winner == self.human else "AI wins!"
            self.status_msg = f"{who}   Press R to play again."
        elif self.ai_thinking:
            self.status_msg = "AI is thinking …"
        elif self.game.current_player == self.human:
            self.status_msg = "Your turn  –  click a cell."
        else:
            self.status_msg = "AI is thinking …"

    # ------------------------------------------------------------------
    # AI move (runs in background thread)
    # ------------------------------------------------------------------

    def _trigger_ai_move(self) -> None:
        if self.ai_agent is None:
            # No model loaded – AI plays random
            import random
            action = random.choice(self.game.legal_actions())
            self._apply_ai_move(action)
            return

        self.ai_thinking = True
        self._update_status()

        game_snapshot = self.game.clone()
        game_id = self.game_id

        def worker() -> None:
            try:
                action = self.ai_agent.choose_action(game_snapshot)
                pygame.event.post(
                    pygame.event.Event(
                        AI_MOVE_EVENT,
                        {"action": action, "game_id": game_id, "error": None}
                    )
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                pygame.event.post(
                    pygame.event.Event(
                        AI_MOVE_EVENT,
                        {"action": None, "game_id": game_id, "error": str(e)}
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def _apply_ai_move(self, action: int) -> None:
        self.ai_thinking = False
        if self.winner is not None:
            return
        if self.game.current_player != self.ai_player:
            return
        if not self.game.is_legal(action):
            return

        self.winner = self.game.step(action)
        if self.ai_agent is not None:
            self.ai_agent.advance(action, self.game)
        self._update_status()

    # ------------------------------------------------------------------
    # Human move
    # ------------------------------------------------------------------

    def _handle_click(self, px: int, py: int) -> None:
        if self.winner is not None:
            return
        if self.game.current_player != self.human:
            return

        cell = self._pixel_to_cell(px, py)
        if cell is None:
            return
        if not self.game.is_legal(cell):
            return

        self.winner = self.game.step(cell)
        if self.ai_agent is not None:
            self.ai_agent.advance(cell, self.game)
        self._update_status()

        if self.winner is None and self.game.current_player == self.ai_player:
            self._trigger_ai_move()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        x = MARGIN + col * CELL_SIZE
        y = MARGIN + row * CELL_SIZE
        return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

    def _cell_centre(self, row: int, col: int) -> tuple[int, int]:
        r = self._cell_rect(row, col)
        return r.centerx, r.centery

    def _pixel_to_cell(self, px: int, py: int) -> Optional[int]:
        col = (px - MARGIN) // CELL_SIZE
        row = (py - MARGIN) // CELL_SIZE
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row * self.cols + col
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_cells()
        self._draw_info_bar()
        pygame.display.flip()

    def _draw_grid(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                rect = self._cell_rect(r, c)
                pygame.draw.rect(self.screen, CELL_EMPTY, rect.inflate(-3, -3), border_radius=6)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, width=1, border_radius=6)

    def _draw_cells(self) -> None:
        legal_set = (
            set(self.game.legal_actions())
            if self.winner is None and self.game.current_player == self.human
            else set()
        )

        for i in range(self.game.n):
            r, c = divmod(i, self.cols)
            rect  = self._cell_rect(r, c)
            inner = rect.inflate(-6, -6)
            cx, cy = self._cell_centre(r, c)
            v = self.game.board[i]

            # Hover highlight on legal cells
            if i == self.hover_cell and i in legal_set:
                self.screen.blit(self.hover_surf, inner.topleft)

            if v == EMPTY:
                continue

            is_p1   = v > 0
            count   = abs(v)
            color   = P1_COLOR  if is_p1 else P2_COLOR
            dark    = P1_DARK   if is_p1 else P2_DARK
            glow    = P1_GLOW   if is_p1 else P2_GLOW
            cap     = self.game.cap[i]

            # Draw each orb sub-circle
            positions = orb_positions(min(count, 4), cx, cy)
            for ox, oy in positions:
                # Glow ring
                pygame.draw.circle(self.screen, glow,  (ox, oy), ORB_RADIUS + 3)
                # Dark border
                pygame.draw.circle(self.screen, dark,  (ox, oy), ORB_RADIUS)
                # Main fill
                pygame.draw.circle(self.screen, color, (ox, oy), ORB_RADIUS - 3)
                # Specular highlight
                pygame.draw.circle(self.screen, glow,  (ox - 4, oy - 4), 4)

            # Orb count label (only when count > 4 which is rare but possible)
            if count > 4:
                lbl = self.font_sm.render(str(count), True, (255, 255, 255))
                self.screen.blit(lbl, lbl.get_rect(center=(cx, cy)))

            # Critical cell indicator: pulsing ring when one orb away from exploding
            if count == cap - 1:
                pygame.draw.circle(self.screen, glow, (cx, cy),
                                   CELL_SIZE // 2 - 6, width=2)

    def _draw_info_bar(self) -> None:
        bar_y = MARGIN + self.rows * CELL_SIZE
        bar_rect = pygame.Rect(0, bar_y, self.win_w, INFO_HEIGHT)
        pygame.draw.rect(self.screen, ( 25, 28, 42), bar_rect)
        pygame.draw.line(self.screen, GRID_COLOR, (0, bar_y), (self.win_w, bar_y), 1)

        # Player labels
        human_lbl = "YOU  (Blue)" if self.human == P1 else "YOU  (Red) "
        ai_lbl    = "AI   (Red) " if self.human == P1 else "AI   (Blue)"
        h_color   = P1_COLOR if self.human == P1 else P2_COLOR
        a_color   = P2_COLOR if self.human == P1 else P1_COLOR

        h_surf = self.font_md.render(human_lbl, True, h_color)
        a_surf = self.font_md.render(ai_lbl,    True, a_color)
        self.screen.blit(h_surf, (MARGIN, bar_y + 12))
        self.screen.blit(a_surf, (MARGIN, bar_y + 38))

        # Cell counts
        h_cells = self.game.alive[self.human]
        a_cells = self.game.alive[self.ai_player]
        cnt_surf = self.font_sm.render(
            f"Cells  You: {h_cells:2d}   AI: {a_cells:2d}", True, DIM_COLOR
        )
        self.screen.blit(cnt_surf, (MARGIN, bar_y + 62))

        # Status message (centred)
        msg_surf = self.font_lg.render(self.status_msg, True, TEXT_COLOR)
        msg_rect = msg_surf.get_rect(center=(self.win_w // 2, bar_y + INFO_HEIGHT // 2))
        self.screen.blit(msg_surf, msg_rect)

        # Key hint
        hint = self.font_sm.render("[R] New game   [ESC] Quit", True, DIM_COLOR)
        self.screen.blit(hint, hint.get_rect(bottomright=(self.win_w - MARGIN, bar_y + INFO_HEIGHT - 8)))

        # Win overlay
        if self.winner is not None:
            overlay = pygame.Surface((self.win_w, MARGIN + self.rows * CELL_SIZE), pygame.SRCALPHA)
            overlay.fill(WIN_OVERLAY)
            self.screen.blit(overlay, (0, 0))

            winner_is_human = self.winner == self.human
            big_msg = "You Win! 🎉" if winner_is_human else "AI Wins!"
            big_color = P1_GLOW if winner_is_human else P2_GLOW
            big_font = pygame.font.SysFont(FONT_NAME, 52, bold=True)
            big_surf = big_font.render(big_msg, True, big_color)
            self.screen.blit(big_surf, big_surf.get_rect(center=(self.win_w // 2,
                             (MARGIN + self.rows * CELL_SIZE) // 2)))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            self.clock.tick(FPS)

            mouse_pos = pygame.mouse.get_pos()
            self.hover_cell = self._pixel_to_cell(*mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_r:
                        self._new_game()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(*event.pos)



                elif event.type == AI_MOVE_EVENT:

                    if getattr(event, "game_id", None) != self.game_id:
                        continue

                    if getattr(event, "error", None):
                        self.ai_thinking = False

                        self.status_msg = f"AI error: {event.error}"

                        continue

                    self._apply_ai_move(event.action)

                    if self.winner is None and self.game.current_player == self.ai_player:
                        self._trigger_ai_move()

            self._draw()


# =============================================================================
# Entry point
# =============================================================================

# Custom pygame event type for thread-safe AI move delivery
AI_MOVE_EVENT = pygame.USEREVENT + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chain Reaction – Human vs AI")
    p.add_argument("--sims",  type=int,   default=CFG.mcts_simulations,
                   help="MCTS simulations per AI move (default: %(default)s)")
    p.add_argument("--time",  type=float, default=CFG.eval_time_limit_s,
                   help="Time budget per AI move in seconds (default: %(default)s). "
                        "Overrides --sims as a soft cap.")
    side = p.add_mutually_exclusive_group()
    side.add_argument("--human-first", dest="human_first", action="store_true",
                      default=True,  help="Human plays as P1 / goes first (default)")
    side.add_argument("--ai-first",    dest="human_first", action="store_false",
                      help="AI plays as P1 / goes first; human plays as P2")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model ──────────────────────────────────────────────────────
    model = load_best_model(device)

    if model is None:
        print(
            "\n[WARNING] No trained model found in "
            f"'{CFG.checkpoint_dir}/'.\n"
            "          Starting without a model – AI will play randomly.\n"
            "          Train first with:  python main.py\n"
        )
        ai_agent = None
    else:
        ai_agent = ModelAgent(
            model,
            device,
            num_simulations=args.sims,
            time_limit_s=args.time,
        )
        sims_label = f"up to {args.sims} sims"
        time_label = f"{args.time:.2f}s/move" if args.time is not None else "no time cap"
        print(f"AI settings: {sims_label}, {time_label}")

    human_player = P1 if args.human_first else P2
    side_label   = "P1 (Blue, goes first)" if human_player == P1 else "P2 (Red, goes second)"
    print(f"You play as: {side_label}")

    # Launch UI ───────────────────────────────────────────────────────
    ui = GameUI(
        ai_agent=ai_agent,
        human_player=human_player,
        num_sims=args.sims,
        time_limit=args.time,
    )
    ui.run()


if __name__ == "__main__":
    main()
