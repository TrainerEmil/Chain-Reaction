"""
test_backend.py – Schnelltest für das Backend (ohne Postman).

Voraussetzung: Server läuft auf localhost:8000
  uvicorn app:app --port 8000

Ausführen:
  python test_backend.py
"""

import json
import sys

try:
    import requests
except ImportError:
    print("requests nicht installiert.  Bitte: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"
SEP  = "─" * 60


def pretty(label: str, data: dict) -> None:
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(json.dumps(data, indent=2, ensure_ascii=False))


# 1. Status
r = requests.get(f"{BASE}/status")
r.raise_for_status()
pretty("GET /status", r.json())

# 2. Neues Spiel
r = requests.post(f"{BASE}/new-game", json={"human_player": 1})
r.raise_for_status()
ng = r.json()
pretty("POST /new-game  (human = P1)", ng)

session_id = ng["state"]["session_id"]
rows = ng["state"]["rows"]
cols = ng["state"]["cols"]

# 3. Ersten legalen Zug finden und ausführen
board = ng["state"]["board"]
cap   = ng["state"]["cap"]
action = next(i for i, v in enumerate(board) if v == 0)  # erster leerer Zelle

print(f"\n  Wähle Zug: cell {action}  (row={action // cols}, col={action % cols})")

r = requests.post(f"{BASE}/move", json={"session_id": session_id, "action": action})
r.raise_for_status()
mv = r.json()
pretty("POST /move", mv)

# 4. Zustand abrufen
r = requests.get(f"{BASE}/state/{session_id}")
r.raise_for_status()
pretty("GET /state/{id}", r.json())

# 5. Illegalen Zug testen (gegnerische Zelle)
state = mv["state"]
board = state["board"]
cp    = state["current_player"]

opp_cell = next((i for i, v in enumerate(board) if v != 0 and v * cp < 0), None)
if opp_cell is not None:
    print(f"\n  Teste illegalen Zug auf gegnerische Zelle {opp_cell} …")
    r = requests.post(f"{BASE}/move", json={"session_id": session_id, "action": opp_cell})
    r.raise_for_status()
    bad = r.json()
    pretty("POST /move (illegal – erwartet valid=false)", bad)
    assert not bad["valid"], "Erwartete valid=false bei illegalem Zug!"
    print("  ✓ Illegaler Zug korrekt abgelehnt")

# 6. Session löschen
r = requests.delete(f"{BASE}/game/{session_id}")
r.raise_for_status()
pretty("DELETE /game/{id}", r.json())

print(f"\n{SEP}")
print("  ✓  Alle Tests bestanden!")
print(SEP)
