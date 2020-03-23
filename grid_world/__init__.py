import numpy as np

width = 4
height = 4
num_states = width * height
S = np.arange(num_states)
A = np.arange(4)  # 0: left, 1: Right, 2: Up, 3: Down
T = np.array([width - 1, num_states - 1])
P = np.zeros((len(S), len(A), len(S), 2))

for s in S:
    if (s % width) == 0:
        P[s, 0, s, 0] = 1.0
    else:
        P[s, 0, s - 1, 0] = 1.0
    if (s + 1) % width == 0:
        P[s, 1, s, 0] = 1.0
    else:
        P[s, 1, s + 1, 0] = 1.0
    if s < width:
        P[s, 2, s, 0] = 1.0
    else:
        P[s, 2, s - width, 0] = 1.0
    if s >= (num_states - width):
        P[s, 3, s, 0] = 1.0
    else:
        P[s, 3, s + width, 0] = 1.0

P[width - 1, :, :, 0] = 0.0
P[num_states - 1, :, :, 0] = 0.0

P[:, :, width - 1, 1] = -5.0
P[:, :, num_states - 1, 1] = 1.0
