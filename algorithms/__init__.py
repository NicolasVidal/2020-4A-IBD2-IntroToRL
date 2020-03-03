import numpy as np


def iterative_policy_evaluation(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        Pi: np.ndarray,
        gamma: float = 0.9,
        theta: float = 0.00001
) -> np.ndarray:
    assert theta > 0
    assert 0 <= gamma <= 1
    V = np.random.random((S.shape[0],))
    V[T] = 0.0
    while True:
        delta = 0
        for s in S:
            v_temp = V[s]
            new_v = 0
            for a in A:
                for s_p in S:
                    new_v += Pi[s, a] * P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
            V[s] = new_v
            delta = np.maximum(delta, np.abs(v_temp - new_v))
        if delta < theta:
            break
    return V
