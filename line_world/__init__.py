import numpy as np

num_states = 5
S = np.arange(num_states)
A = np.array([0, 1])  # 0: left, 1 : right
T = np.array([0, num_states - 1])
P = np.zeros((len(S), len(A), len(S), 2))

for s in S[1:-1]:
    P[s, 0, s - 1, 0] = 1.0
    P[s, 1, s + 1, 0] = 1.0
P[1, 0, 0, 1] = -1.0
P[num_states - 2, 1, num_states - 1, 1] = 1.0


def reset() -> int:
    return num_states // 2


def is_terminal(state: int) -> bool:
    return state in T


def step(state: int, action: int) -> (int, float, bool):
    assert not is_terminal(state)
    next_state = np.random.choice(S, p=P[state, action, :, 0])
    reward = P[state, action, next_state, 1]
    return next_state, reward, is_terminal(state)
