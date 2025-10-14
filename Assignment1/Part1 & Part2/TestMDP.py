from MDP import *

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
# a_0 -> A (Advertising), a_1 -> S (Saving money)
# s_0 -> PU (Poor Unknown), s_1 -> PF (Poor Famous), s_2 -> RU (Rich Unkown), s_3 -> RF (Rich Famous)
# ex: 
# T[0][0][0] = 0.5 -> cur_a = 0, cur_s = 0, 50% transmit to s_0
# T[0][0][1] = 0.5 -> cur_a = 0, cur_s = 0, 50% transmit to s_1
# T[0][0][2] = 0   -> cur_a = 0, cur_s = 0, 50% transmit to s_2
# T[0][0][3] = 0   -> cur_a = 0, cur_s = 0, 50% transmit to s_3
T = np.array([[[0.5, 0.5, 0, 0],  
               [0, 1, 0, 0],
               [0.5, 0.5, 0, 0],
               [0, 1, 0, 0]],

              [[1, 0, 0, 0],
               [0.5, 0, 0, 0.5],
               [0.5, 0, 0.5, 0],
               [0, 0, 0.5, 0.5]]])

# Reward function: |A| x |S| array
# R[cur_action, cur_state]
R = np.array([[0, 0, 10, 10],
              [0, 0, 10, 10]])

# Discount factor: scalar in [0,1)
discount = 0.9        

# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
print(f"========================= by Value Iteration =========================")
[V, nIterations, epsilon] = mdp.valueIteration(initialV= np.zeros(mdp.nStates))
policy = mdp.extractPolicy(V)
print(f"epsilon = {epsilon}")
print(f"Policy = {policy}")
print(f"V = {V}\nnIterations = {nIterations}")

print(f"========================= by Policy Iteration =========================")
[policy, V, iterId] = mdp.policyIteration(initialPolicy= np.array([1, 0, 0, 0]))
print(f"Policy = {policy}")
print(f"V = {V}\nnIterations = {iterId}")

print(f"========================= by Truncated Policy Iteration =========================")
# [V, iterId, epsilon] = mdp.evaluatePolicyPartially(policy= np.array([1, 0, 1, 0]), initialV= np.array([0, 10, 0, 13]))
[policy, V, iterId, tolerance] = mdp.modifiedPolicyIteration(initialPolicy= np.array([1, 0, 1, 0]), initialV= np.array([0, 10, 0, 13]))
print(f"policy = {policy}")
print(f"V = {V}\nnIterations = {iterId}") 


