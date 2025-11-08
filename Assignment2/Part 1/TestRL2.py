import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Bernoulli distribution:
    x-axis : 0 or 1
    y-axis : probability of 0 and 1
    ex: p(0) = 0.5, p(1) = 0.5
    Mean[x] = 0*p(0) + 1*p(1) = 0*0.5 + 0.5*1 = 0.5 = p(1)

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
# shape (nAction, nState) of each matrix

# Transition Probability Matrix T (shape (3, 1, 1))
# no state transition
T = np.array([  
    [[1]],
    [[1]],
    [[1]]
])

# Reward Probability Matrix R (shape (3, 1))
R = np.array([
    [0.3],
    [0.5],
    [0.7]
])

discount = 0.999
mdp = MDP.MDP(T, R, discount)
banditProblem = RL2.RL2(mdp, sampleBernoulli)

# # Test epsilon greedy strategy
# print("\nDoing epsilon-greedy...")
# rewards_earned = banditProblem.epsilonGreedyBandit(nIterations= 200)
# print("\nFinish epsilonGreedyBandit")
# plt.figure(0)
# plt.clf()
# plt.xlabel("iterations")
# plt.ylabel("avg. reward")
# plt.title("$\epsilon$-greedy")
# plt.plot(rewards_earned)
# plt.savefig('/home/super_trumpet/NCKU/Homework/114-2-DRL/Assignment2/figures/test/epsilon-greedy')



# # Test Thompson sampling strategy
# print("\nDoing thompsonSampling...")
# rewards_earned = banditProblem.thompsonSamplingBandit(prior= np.ones([mdp.nActions, 2]), nIterations= 200)
# print("\nFinish thompsonSamplingBandit")
# plt.figure(1)
# plt.clf()
# plt.xlabel("iterations")
# plt.ylabel("avg. reward")
# plt.title("thompson_Sampling")
# plt.plot(rewards_earned)
# plt.savefig('/home/super_trumpet/NCKU/Homework/114-2-DRL/Assignment2/figures/test/thompson_Sampling')

# # Test UCB strategy
# print("\nDoing UCB...")
# rewards_earned = banditProblem.UCBbandit(nIterations= 200)
# print("\nFinish UCBbandit")
# plt.figure(2)
# plt.clf()
# plt.xlabel("iterations")
# plt.ylabel("avg. reward")
# plt.title("UCB")
# plt.plot(rewards_earned)
# plt.savefig('/home/super_trumpet/NCKU/Homework/114-2-DRL/Assignment2/figures/test/UCBbandit')



#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9        
mdp = MDP.MDP(T, R, discount)
rlProblem = RL2.RL2(mdp, np.random.normal)

# Test model-based RL
[V, policy, cumulative_discounted_reward_episode] = rlProblem.modelBasedRL(
    s0= 0, 
    defaultT= np.ones([mdp.nActions, mdp.nStates, mdp.nStates]) / mdp.nStates,  # uniform distribution
    initialR= np.zeros([mdp.nActions, mdp.nStates]), 
    nEpisodes= 100, 
    nSteps= 100, 
    epsilon= 0.05)
print("\nmodel-based RL results")
print(f"v = {V}")
print(f"policy = {policy}")

plt.figure(0)
plt.clf()
plt.plot(cumulative_discounted_reward_episode)
plt.xlabel("episodes")
plt.ylabel("cumulative discounted reward")
plt.title("model-based RL")
plt.savefig('/home/super_trumpet/NCKU/Homework/114-2-DRL/Assignment2/figures/test/model-based RL')