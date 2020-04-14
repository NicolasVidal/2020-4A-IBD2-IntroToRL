from algorithms import off_policy_monte_carlo_control
from grid_world import is_terminal, step, S, A, reset

if __name__ == "__main__":
    Q, Pi = off_policy_monte_carlo_control(len(S), len(A),
                                           reset,
                                           step,
                                           is_terminal,
                                           max_episodes=1000, max_steps_per_episode=100)
    print(Q)
    print(Pi)

    Q, Pi = off_policy_monte_carlo_control(len(S), len(A),
                                           reset,
                                           step,
                                           is_terminal,
                                           max_episodes=1000, max_steps_per_episode=100,
                                           epsilon=0.2,
                                           epsilon_greedy_behaviour_policy=True)
    print(Q)
    print(Pi)
