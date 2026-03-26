"""
model_loader.py – Findet und lädt das beste verfügbare Modell.

Suchreihenfolge:
  1. checkpoints/model_final.pt          (letztes fertiges Modell)
  2. checkpoints/training_state.pt       (laufender Trainingsstand → best_model)
  3. checkpoints/model_iter_NNN.pt       (höchste Iteration)
  4. Fallback: frisch initialisiertes Netz (zufällig)
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import torch

from config import CFG


def _find_best_checkpoint() -> Optional[str]:
    """Gibt den Pfad zum besten verfügbaren Checkpoint zurück, oder None."""
    base = CFG.checkpoint_dir

    # 1. fertiges Modell
    final = os.path.join(base, "model_final.pt")
    if os.path.exists(final):
        return final

    # 2. training_state.pt  → enthält best_model_state
    state = os.path.join(base, "training_state.pt")
    if os.path.exists(state):
        return state

    # 3. model_iter_NNN.pt  → nimm die höchste Iteration
    pattern = os.path.join(base, "model_iter_*.pt")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return candidates[-1]

    return None


def load_best_model(device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, str]:
    """
    Lädt das beste Modell und gibt (model, source_description) zurück.

    Falls kein Checkpoint vorhanden ist, wird ein frisch initialisiertes
    Netz zurückgegeben (nützlich für Entwicklung / Tests).
    """
    # Importiere erst hier, damit model_loader ohne model.py lauffähig bleibt.
    from model import build_model  # type: ignore

    model, dev = build_model(device)
    if device is None:
        device = dev

    checkpoint_path = _find_best_checkpoint()

    if checkpoint_path is None:
        model.eval()
        return model, "random (no checkpoint found)"

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # training_state.pt hat best_model_state, reine Modell-Dateien haben direkt state_dict
    if "best_model_state" in payload:
        model.load_state_dict(payload["best_model_state"])
        source = f"best_model from {checkpoint_path}"
    elif "model_state" in payload:
        model.load_state_dict(payload["model_state"])
        source = f"model_state from {checkpoint_path}"
    else:
        model.load_state_dict(payload)
        source = checkpoint_path

    model.eval()
    return model, source
