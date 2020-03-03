from algorithms import iterative_policy_evaluation
from grid_world import S, A, T, P
from policies import tabular_random_uniform_policy

if __name__ == "__main__":
    print("Evaluation policy random :")
    Pi = tabular_random_uniform_policy(S.shape[0], A.shape[0])
    V = iterative_policy_evaluation(S, A, P, T, Pi)
    print(V)
