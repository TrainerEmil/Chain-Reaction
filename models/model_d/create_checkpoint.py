"""
Run once from the project root to create the weights file:

    python models/model_d/create_checkpoint.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))      # model_d's own files
sys.path.insert(1, str(Path(__file__).parent.parent.parent))  # project root

import torch
from model import build_model

model, _ = build_model()

out = Path(__file__).parent / "checkpoints" / "model_final.pt"
out.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), out)
print(f"Saved → {out}")