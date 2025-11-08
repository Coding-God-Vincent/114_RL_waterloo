import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Graph Generation
plt.figure()
plt.clf()
plt.title("Part 1-2 : TestBandit")
plt.xlabel("Iterations")
plt.ylabel("avg. reward")

# #----------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Test epsilon greedy strategy
print("\nDoing epsilonGreedyBandit...")
rewards_earned = banditProblem.epsilonGreedyBandit(nIterations= 200)
print("Finish epsilonGreedyBandit")
plt.plot(rewards_earned, label= "$\epsilon$-greedy")


# Test Thompson sampling strategy
print("\nDoing thompsonSamplingBandit...")
rewards_earned = banditProblem.thompsonSamplingBandit(prior= np.ones([mdp.nActions, 2]), nIterations= 200)
print("Finish thompsonSamplingBandit")
plt.plot(rewards_earned, label= "thompson Sampling")

# Test UCB strategy
print("\nDoing UCBbandit...")
rewards_earned = banditProblem.UCBbandit(nIterations= 200)
print("Finish UCBbandit")
plt.plot(rewards_earned, label= "UCB Sampling")

plt.legend()
plt.savefig("/home/super_trumpet/NCKU/Homework/114-2-DRL/Assignment2/Part 1/figures/outcome/Part 1-2 : TestBandit")