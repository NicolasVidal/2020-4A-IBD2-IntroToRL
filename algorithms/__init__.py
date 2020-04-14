from typing import Callable

import numpy as np

from policies import tabular_random_uniform_policy
from utils import step_until_the_end_of_the_episode_and_generate_trajectory


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


def first_visit_monte_carlo_prediction(
        pi: np.ndarray,
        reset_func: Callable,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
        exploring_starts: bool = False
) -> np.ndarray:
    states_count = pi.shape[0]
    states = np.arange(states_count)
    V = np.random.random(states_count)

    for s in states:
        if is_terminal_func(s):
            V[s] = 0.0

    returns = np.zeros(states_count)
    returns_count = np.zeros(states_count)

    for episode_id in range(max_episodes):
        s0 = np.random.choice(states) if exploring_starts else reset_func()
        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_generate_trajectory(s0, pi, step_func,
                                                                                              is_terminal_func,
                                                                                              max_steps_per_episode)
        G = 0.0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            if st in s_list[0:t]:
                continue
            returns[st] += G
            returns_count[st] += 1
            V[st] = returns[st] / returns_count[st]
    return V


def monte_carlo_with_exploring_starts_control(
        states_count: int,
        actions_count: int,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99
) -> (np.ndarray, np.ndarray):
    pi = tabular_random_uniform_policy(states_count, actions_count)
    states = np.arange(states_count)
    actions = np.arange(actions_count)

    Q = np.random.random((states_count, actions_count))

    for s in states:
        if is_terminal_func(s):
            Q[s, :] = 0.0
            pi[s, :] = 0.0

    returns = np.zeros((states_count, actions_count))
    returns_count = np.zeros((states_count, actions_count))

    for episode_id in range(max_episodes):
        s0 = np.random.choice(states)

        if is_terminal_func(s0):
            episode_id -= 1
            continue

        a0 = np.random.choice(actions)

        s1, r1, terminal = step_func(s0, a0)

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_generate_trajectory(s1, pi, step_func,
                                                                                              is_terminal_func,
                                                                                              max_steps_per_episode)
        s_list.insert(0, s0)
        a_list.insert(0, a0)
        r_list.insert(0, r1)

        G = 0.0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            if (st, at) in zip(s_list[0:t], a_list[0:t]):
                continue

            returns[st, at] += G
            returns_count[st, at] += 1
            Q[st, at] = returns[st, at] / returns_count[st, at]
            pi[st, :] = 0.0
            pi[st, np.argmax(Q[st, :])] = 1.0
    return Q, pi


def on_policy_first_visit_monte_carlo_epsilon_soft_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
        epsilon: float = 0.1
) -> (np.ndarray, np.ndarray):
    pi = tabular_random_uniform_policy(states_count, actions_count)
    states = np.arange(states_count)
    actions = np.arange(actions_count)

    Q = np.random.random((states_count, actions_count))

    for s in states:
        if is_terminal_func(s):
            Q[s, :] = 0.0
            pi[s, :] = 0.0

    returns = np.zeros((states_count, actions_count))
    returns_count = np.zeros((states_count, actions_count))

    for episode_id in range(max_episodes):
        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_generate_trajectory(s0, pi, step_func,
                                                                                              is_terminal_func,
                                                                                              max_steps_per_episode)

        G = 0.0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            if (st, at) in zip(s_list[0:t], a_list[0:t]):
                continue

            returns[st, at] += G
            returns_count[st, at] += 1
            Q[st, at] = returns[st, at] / returns_count[st, at]
            pi[st, :] = epsilon / actions_count
            pi[st, np.argmax(Q[st, :])] = 1.0 - epsilon + epsilon / actions_count
    return Q, pi


def off_policy_monte_carlo_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_greedy_behaviour_policy: bool = False
) -> (np.ndarray, np.ndarray):
    b = tabular_random_uniform_policy(states_count, actions_count)
    pi = tabular_random_uniform_policy(states_count, actions_count)
    states = np.arange(states_count)

    Q = np.random.random((states_count, actions_count))

    for s in states:
        if is_terminal_func(s):
            Q[s, :] = 0.0
            pi[s, :] = 0.0
        pi[s, :] = 0
        pi[s, np.argmax(Q[s, :])] = 1.0

    C = np.zeros((states_count, actions_count))

    for episode_id in range(max_episodes):
        if epsilon_greedy_behaviour_policy:
            for s in states:
                b[s, :] = epsilon / actions_count
                b[s, np.argmax(Q[s, :])] = 1.0 - epsilon + epsilon / actions_count

        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_generate_trajectory(s0, b, step_func,
                                                                                              is_terminal_func,
                                                                                              max_steps_per_episode)

        G = 0.0
        W = 1
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            C[st, at] += W

            Q[st, at] += W / C[st, at] * (G - Q[st, at])
            pi[st, :] = 0
            pi[st, np.argmax(Q[st, :])] = 1.0
            if np.argmax(Q[st, :]) != at:
                break
            W = W / b[st, at]
    return Q, pi


def tabular_td_zero_prediction(
        pi: np.ndarray,
        reset_func: Callable,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
        alpha: float = 0.1
) -> np.ndarray:
    states_count = pi.shape[0]
    actions_count = pi.shape[1]

    states = np.arange(states_count)
    actions = np.arange(actions_count)
    V = np.random.random(states_count)

    for s in states:
        if is_terminal_func(s):
            V[s] = 0.0

    for episode_id in range(max_episodes):
        s = reset_func()
        step = 0
        while not is_terminal_func(s) and step < max_steps_per_episode:
            a = np.random.choice(actions, p=pi[s])
            (s_p, r, t) = step_func(s, a)
            V[s] += alpha * (r + gamma * V[s_p] - V[s])
            s = s_p
            step += 1
    return V


def tabular_sarsa_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
        alpha: float = 0.01,
        epsilon: float = 0.75
) -> (np.ndarray, np.ndarray):
    states = np.arange(states_count)
    actions = np.arange(actions_count)

    Q = np.random.random((states_count, actions_count))
    pi = np.random.random((states_count, actions_count))

    for s in states:
        if is_terminal_func(s):
            Q[s, :] = 0.
            pi[s, :] = 0.

    for episode_id in range(max_episodes):
        s = reset_func()
        step = 0
        rdm = np.random.random()
        a = np.random.choice(actions) if rdm < epsilon else np.argmax(Q[s, :])

        while not is_terminal_func(s) and step < max_steps_per_episode:
            (s_p, r, t) = step_func(s, a)
            rdm = np.random.random()
            a_p = np.random.choice(actions) if rdm < epsilon else np.argmax(Q[s_p, :])
            Q[s, a] += alpha * (r + gamma * Q[s_p, a_p] - Q[s, a])
            s = s_p
            a = a_p
            step += 1

    for s in states:
        if is_terminal_func(s):
            pi[s, :] = 0.
        pi[s, :] = epsilon / actions_count
        pi[s, np.argmax(Q[s, :])] = 1 - epsilon + epsilon / actions_count

    return Q, pi


def tabular_q_learning_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
        alpha: float = 0.01,
        epsilon: float = 0.75
) -> (np.ndarray, np.ndarray):
    states = np.arange(states_count)
    actions = np.arange(actions_count)

    Q = np.random.random((states_count, actions_count))
    pi = np.random.random((states_count, actions_count))

    for s in states:
        if is_terminal_func(s):
            Q[s, :] = 0.
            pi[s, :] = 0.

    for episode_id in range(max_episodes):
        s = reset_func()
        step = 0

        while not is_terminal_func(s) and step < max_steps_per_episode:
            rdm = np.random.random()
            a = np.random.choice(actions) if rdm < epsilon else np.argmax(Q[s, :])
            (s_p, r, t) = step_func(s, a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_p, :]) - Q[s, a])
            s = s_p
            step += 1

    for s in states:
        if is_terminal_func(s):
            pi[s, :] = 0.
        pi[s, :] = 0.0
        pi[s, np.argmax(Q[s, :])] = 1.0

    return Q, pi
