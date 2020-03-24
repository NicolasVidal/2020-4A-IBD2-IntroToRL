from algorithms import monte_carlo_with_exploring_starts_control
from line_world import is_terminal, step, S, A

if __name__ == "__main__":
    Q, Pi = monte_carlo_with_exploring_starts_control(len(S), len(A), step, is_terminal,
                                                      max_episodes=10000, max_steps_per_episode=100)
    print(Q)
    print(Pi)
