from algorithms import policy_iteration
from line_world import S, A, T, P

if __name__ == "__main__":
    V, Pi = policy_iteration(S, A, P, T)
    print(V)
    print(Pi)
