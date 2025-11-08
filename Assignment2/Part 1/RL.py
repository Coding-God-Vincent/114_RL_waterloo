import numpy as np
import MDP
from tqdm import tqdm

class RL:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards, it's a probability distribution (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r) -> in this example, it's a normal distribution
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed
 
        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action, state])
        # 回傳一個一樣 shape 的 np.array。第 i 格會是 0~i 格數字的總和
        # ex: np.cumsum([1, 2, 3]) -> [1, 3, 6]
        cumProb = np.cumsum(self.mdp.T[action, state, :])  
        # ex: T[0, 0, :] = [0.5, 0.5, 0, 0]
        # np.cumsum -> [0.5, 1, 1, 1]
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        # ex: np.random.rand = 0.6 -> [false, true, true, true]
        # np.where()[0] = [0] (true 所在的 index) -> [1, 2, 3] 
        # np.where()[0][0] = 1 -> 為 nextState
        return [reward, nextState]

    def qLearning(self, s0, initialQ, nEpisodes, nSteps, epsilon= 0, temperature= 0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration (explore by softmax)
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        # Q = np.zeros([self.mdp.nActions, self.mdp.nStates])
        # policy = np.zeros(self.mdp.nStates, int)

        cumulative_discounted_reward_total = np.zeros((nEpisodes))  # Cumulative discounted reward of each episode

        execute_times = np.zeros((self.mdp.nActions, self.mdp.nStates))
        Q = initialQ
        for i in range(nEpisodes):
            rewards_each_epsiode = np.zeros(100)  # 100 steps in one episode
            state = s0
            for j in range(nSteps):
                # choosing action by epsilon-greedy
                if np.random.rand() < epsilon: action = np.random.choice(self.mdp.nActions)
                else: action = np.argmax(Q[:, state])
                # update execute_times
                execute_times[action][state] += 1
                # interact with the environment
                [reward, nextState] = self.sampleRewardAndNextState(state= state, action= action)
                # store cumulative reward
                rewards_each_epsiode[j] = reward * np.pow(self.mdp.discount, j)  # r*(gamma)^(j)
                # update Q-Table : Q(s, a) <- Q(s, a) + lr * (reward + \gamma * Q(s', argmax_a{Q(s', a')}) - Q(s, a))
                Q[action][state] = Q[action][state] + (1/execute_times[action][state]) * ((reward + self.mdp.discount * Q[np.argmax(Q[:, nextState])][nextState]) - Q[action][state])
                state = nextState
            cumulative_discounted_reward_total[i] = np.sum(rewards_each_epsiode)  # cumulative discounted reward of ith epsiode
        # extract policy from trained Q-table
        policy = [int(np.argmax(Q[:, s])) for s in range(self.mdp.nStates)]

        

        return [Q, policy, cumulative_discounted_reward_total]    