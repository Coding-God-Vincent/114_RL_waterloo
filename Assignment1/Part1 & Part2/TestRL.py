import numpy as np
import MDP
import RL


''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
T = np.array([[[0.5 ,0.5, 0, 0],
               [0, 1, 0, 0],
               [0.5, 0.5, 0, 0],
               [0, 1, 0, 0]],

              [[1, 0, 0 ,0],
               [0.5, 0, 0, 0.5],
               [0.5, 0, 0.5, 0],
               [0, 0, 0.5, 0.5]]])

R = np.array([[0, 0, 10, 10],
              [0, 0, 10, 10]])

discount = 0.9        
mdp = MDP.MDP(T, R, discount)
rlProblem = RL.RL(mdp, np.random.normal)  
# nlp.random.normal 會是一個物件，代表一個機率分布，可以透過參數對其進行取樣。
# ex: np.random.normal(loc= "平均值", scale= "標準差", size: "輸出的值的 size")

# Test Q-learning 
[Q, policy] = rlProblem.qLearning(s0= 0, initialQ= np.zeros([mdp.nActions, mdp.nStates]), nEpisodes= 1000, nSteps= 100, epsilon= 0.3)
print("\nQ-learning results")
print(Q)
print(policy)