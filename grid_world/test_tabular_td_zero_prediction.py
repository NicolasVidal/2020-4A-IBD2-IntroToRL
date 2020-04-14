import numpy as np

from algorithms import tabular_td_zero_prediction
from grid_world import is_terminal, step, reset, S, A
from policies import tabular_random_uniform_policy

if __name__ == "__main__":
    print("Evaluation policy random :")
    Pi = tabular_random_uniform_policy(S.shape[0], A.shape[0])
    V = tabular_td_zero_prediction(Pi, reset, step, is_terminal, alpha=0.01,
                                   max_episodes=20000, max_steps_per_episode=100)
    print(V)
