import numpy as np

from algorithms import first_visit_monte_carlo_prediction
from line_world import is_terminal, step, reset, S, A
from policies import tabular_random_uniform_policy

if __name__ == "__main__":
    print("Evaluation policy random :")
    Pi = tabular_random_uniform_policy(S.shape[0], A.shape[0])
    V = first_visit_monte_carlo_prediction(Pi, reset, step, is_terminal,
                                           max_episodes=10000, max_steps_per_episode=100)
    print(V)

    print('Evaluation Policy "Toujours vers la droite !" :')
    Pi = np.zeros((S.shape[0], A.shape[0]))
    Pi[1:-1, 1] = 1.0
    V = first_visit_monte_carlo_prediction(Pi, reset, step, is_terminal,
                                           exploring_starts=True)
    print(V)
