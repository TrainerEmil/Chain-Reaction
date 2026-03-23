"""
tests/test_all.py – Core sanity checks.

Run with:
  python -m pytest tests/ -v
  or directly:
  python tests/test_all.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from config import CFG
from engine import ChainReaction, P1, P2
from encoding import encode_state, legal_action_mask, encode_state_tensor
from model import ChainReactionNet, build_model
from replay_buffer import ReplayBuffer
from mcts import MCTS
from selfplay import play_game


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_game(rows=CFG.rows, cols=CFG.cols):
    g = ChainReaction(rows, cols)
    g.reset()
    return g


# ---------------------------------------------------------------------------
# encoding.py
# ---------------------------------------------------------------------------

class TestEncoding:

    def test_encode_state_shape(self):
        game = make_game()
        enc = encode_state(game)
        assert enc.shape == (CFG.num_channels, CFG.rows, CFG.cols), \
            f"Expected ({CFG.num_channels}, {CFG.rows}, {CFG.cols}), got {enc.shape}"

    def test_encode_state_dtype(self):
        game = make_game()
        enc = encode_state(game)
        assert enc.dtype == np.float32

    def test_encode_state_values_non_negative(self):
        game = make_game()
        game.step(0)
        game.step(1)
        enc = encode_state(game)
        assert (enc >= 0).all(), "Encoded values must be non-negative"

    def test_legal_mask_length(self):
        game = make_game()
        mask = legal_action_mask(game)
        assert len(mask) == game.n, \
            f"Mask length {len(mask)} != n {game.n}"

    def test_legal_mask_all_legal_at_start(self):
        game = make_game()
        mask = legal_action_mask(game)
        assert mask.sum() == game.n, "All cells should be legal at game start"

    def test_legal_mask_respects_opponent_cells(self):
        game = make_game()
        # P1 places on cell 0, P2 places on cell 1
        game.step(0)
        game.step(1)
        # Now it's P1's turn; cell 1 belongs to P2 → illegal
        mask = legal_action_mask(game)
        assert mask[0] == 1.0, "Cell 0 (P1's) should be legal for P1"
        assert mask[1] == 0.0, "Cell 1 (P2's) should be illegal for P1"

    def test_encode_state_tensor_shape(self):
        game = make_game()
        t = encode_state_tensor(game)
        assert t.shape == (1, CFG.num_channels, CFG.rows, CFG.cols)


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------

class TestModel:

    def test_forward_output_shapes(self):
        model, device = build_model()
        model.eval()
        x = torch.zeros(2, CFG.num_channels, CFG.rows, CFG.cols).to(device)
        with torch.no_grad():
            policy_logits, value = model(x)
        assert policy_logits.shape == (2, CFG.rows * CFG.cols), \
            f"Policy shape wrong: {policy_logits.shape}"
        assert value.shape == (2, 1), \
            f"Value shape wrong: {value.shape}"

    def test_value_in_range(self):
        model, device = build_model()
        model.eval()
        x = torch.randn(4, CFG.num_channels, CFG.rows, CFG.cols).to(device)
        with torch.no_grad():
            _, value = model(x)
        assert (value >= -1.0).all() and (value <= 1.0).all(), \
            "Value must be in [-1, 1] (tanh output)"


# ---------------------------------------------------------------------------
# replay_buffer.py
# ---------------------------------------------------------------------------

class TestReplayBuffer:

    def _make_example(self, rows=CFG.rows, cols=CFG.cols):
        state   = np.zeros((CFG.num_channels, rows, cols), dtype=np.float32)
        policy  = np.ones(rows * cols, dtype=np.float32) / (rows * cols)
        value   = 1.0
        return (state, policy, value)

    def test_add_and_len(self):
        buf = ReplayBuffer(max_size=100)
        buf.add([self._make_example() for _ in range(10)])
        assert len(buf) == 10

    def test_max_size_respected(self):
        buf = ReplayBuffer(max_size=5)
        buf.add([self._make_example() for _ in range(10)])
        assert len(buf) == 5, "Buffer should not exceed max_size"

    def test_sample_shapes(self):
        buf = ReplayBuffer(max_size=100)
        buf.add([self._make_example() for _ in range(20)])
        states, policies, values = buf.sample(8)
        assert states.shape   == (8, CFG.num_channels, CFG.rows, CFG.cols)
        assert policies.shape == (8, CFG.rows * CFG.cols)
        assert values.shape   == (8,)

    def test_sample_empty_raises(self):
        buf = ReplayBuffer(max_size=100)
        with pytest.raises(RuntimeError):
            buf.sample(4)


# ---------------------------------------------------------------------------
# mcts.py
# ---------------------------------------------------------------------------

class TestMCTS:

    def test_mcts_returns_legal_action(self):
        model, device = build_model()
        mcts = MCTS(model, device)
        game = make_game()
        policy, action = mcts.run(game, num_simulations=8, temperature=1.0)
        assert game.is_legal(action), f"Action {action} is not legal!"

    def test_mcts_policy_sums_to_one(self):
        model, device = build_model()
        mcts = MCTS(model, device)
        game = make_game()
        policy, _ = mcts.run(game, num_simulations=8, temperature=1.0)
        assert abs(policy.sum() - 1.0) < 1e-5, "Policy must sum to 1"

    def test_mcts_policy_shape(self):
        model, device = build_model()
        mcts = MCTS(model, device)
        game = make_game()
        policy, _ = mcts.run(game, num_simulations=8, temperature=1.0)
        assert policy.shape == (game.n,)

    def test_mcts_greedy_is_deterministic(self):
        model, device = build_model()
        mcts = MCTS(model, device)
        game = make_game()
        np.random.seed(0)
        _, a1 = mcts.run(game, num_simulations=16, temperature=0)
        np.random.seed(0)
        _, a2 = mcts.run(game, num_simulations=16, temperature=0)
        assert a1 == a2, "Greedy MCTS should be deterministic with same seed"


# ---------------------------------------------------------------------------
# selfplay.py
# ---------------------------------------------------------------------------

class TestSelfPlay:
    # The model is built for CFG.rows × CFG.cols.
    # play_game MUST use the same dimensions, otherwise the model's
    # FC layers receive a tensor with the wrong number of features.

    def test_play_game_produces_valid_examples(self):
        model, device = build_model()
        examples = play_game(
            model, device,
            rows=CFG.rows, cols=CFG.cols,   # must match the model's board size
            num_simulations=4,
            max_game_length=100,
            seed=0,
        )
        # Game might be empty if it timed out – allow empty
        for state, policy, value in examples:
            assert state.shape[0] == CFG.num_channels
            assert abs(policy.sum() - 1.0) < 1e-5, "Policy target must sum to 1"
            assert value in (-1.0, 1.0), f"Value target must be ±1, got {value}"

    def test_play_game_value_targets_consistent(self):
        """The value targets must not all be the same (both players must be represented)."""
        model, device = build_model()
        examples = play_game(
            model, device,
            rows=CFG.rows, cols=CFG.cols,   # must match the model's board size
            num_simulations=4,
            max_game_length=200,
            seed=1,
        )
        if examples:
            values = [v for _, _, v in examples]
            assert not all(v == values[0] for v in values), \
                "Value targets should include both +1 and -1"


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
