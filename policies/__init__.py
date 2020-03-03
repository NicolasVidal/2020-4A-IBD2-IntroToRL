import numpy as np


def tabular_random_uniform_policy(state_size: int, action_size: int) -> np.ndarray:
    assert action_size > 0
    return np.ones((state_size, action_size,)) / action_size
