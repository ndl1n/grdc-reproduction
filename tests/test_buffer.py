from __future__ import annotations

import numpy as np

from grdc.buffer import BufferShapes, RolloutBuffer


def test_gae_stops_at_episode_boundary() -> None:
    shapes = BufferShapes(2, 2, 3, 4, 5, 6)
    buffer = RolloutBuffer(shapes)
    buffer.reset(n_prototypes=3)
    common = dict(
        observations=np.zeros((2, 3), np.float32),
        state=np.zeros(4, np.float32),
        adjacency=np.zeros((2, 2), bool),
        available_actions=np.ones((2, 5), bool),
        hidden=np.zeros((2, 6), np.float32),
        actions=np.zeros(2, np.int64),
        log_probabilities=np.zeros(2, np.float32),
        role_assignments=np.zeros(2, np.int64),
        role_probabilities=np.full((2, 3), 1 / 3, np.float32),
        value=0.0,
        next_value=0.0,
        episode_start=True,
    )
    buffer.add(**common, reward=1.0, episode_end=True)
    buffer.add(**common, reward=2.0, episode_end=True)
    buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(buffer.advantages, [1.0, 2.0])

