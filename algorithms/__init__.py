import numpy as np

from policies import tabular_random_uniform_policy


def iterative_policy_evaluation(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        Pi: np.ndarray,
        gamma: float = 0.99,
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


def policy_iteration(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        gamma: float = 0.99,
        theta: float = 0.00001
) -> (np.ndarray, np.ndarray):
    Pi = tabular_random_uniform_policy(S.shape[0], A.shape[0])
    while True:
        V = iterative_policy_evaluation(S, A, P, T, Pi, gamma, theta)
        policy_stable = True
        for s in S:
            old_action = np.argmax(Pi[s])
            best_action = 0
            best_action_score = -9999999999
            for a in A:
                tmp_sum = 0
                for s_p in S:
                    tmp_sum += Pi[s, a] * P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
                if tmp_sum > best_action_score:
                    best_action = a
                    best_action_score = tmp_sum
            Pi[s] = 0.0
            Pi[s, best_action] = 1.0
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    V = iterative_policy_evaluation(S, A, P, T, Pi, gamma, theta)
    return V, Pi
