from algorithms import off_policy_monte_carlo_control
from line_world import is_terminal, step, S, A, reset

if __name__ == "__main__":
    Q, Pi = off_policy_monte_carlo_control(len(S), len(A),
                                           reset,
                                           step,
                                           is_terminal,
                                           max_episodes=10000, max_steps_per_episode=100)
    print(Q)
    print(Pi)
