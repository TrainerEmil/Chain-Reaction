"""
replay_buffer.py – Fixed-size circular buffer for (state, policy, value) triples.

Design:
- Uses a Python deque with maxlen so old data is automatically dropped.
- Stores numpy arrays to keep memory usage low.
- Sampling is done without replacement to reduce correlation.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np

# A single training example: (encoded_state, policy_target, value_target)
Example = Tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    """
    Circular buffer for self-play training examples.

    Parameters
    ----------
    max_size : Maximum number of examples to keep.
               When full, the oldest example is dropped on each insert.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._buffer: deque[Example] = deque(maxlen=max_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, examples: List[Example]) -> None:
        """Append a list of examples to the buffer."""
        for ex in examples:
            self._buffer.append(ex)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample *batch_size* examples uniformly at random (with replacement
        if the buffer is smaller than batch_size).

        Returns
        -------
        states   : (B, C, H, W) float32
        policies : (B, n_actions) float32
        values   : (B,)         float32
        """
        if len(self._buffer) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        replace = len(self._buffer) < batch_size
        indices = random.choices(range(len(self._buffer)), k=batch_size) \
                  if replace else \
                  random.sample(range(len(self._buffer)), batch_size)

        states, policies, values = [], [], []
        for idx in indices:
            s, p, v = self._buffer[idx]
            states.append(s)
            policies.append(p)
            values.append(v)

        return (
            np.stack(states, axis=0).astype(np.float32),
            np.stack(policies, axis=0).astype(np.float32),
            np.array(values, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.max_size})"
