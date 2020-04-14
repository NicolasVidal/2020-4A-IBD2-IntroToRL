from algorithms import on_policy_first_visit_monte_carlo_epsilon_soft_control
from grid_world import is_terminal, step, S, A, reset

if __name__ == "__main__":
    Q, Pi = on_policy_first_visit_monte_carlo_epsilon_soft_control(len(S), len(A),
                                                                   reset,
                                                                   step,
                                                                   is_terminal,
                                                                   max_episodes=100000, max_steps_per_episode=100)
    print(Q)
    print(Pi)
