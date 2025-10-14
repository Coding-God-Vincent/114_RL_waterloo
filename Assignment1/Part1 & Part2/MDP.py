import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''
        # assert: determine whether the condition is true, otherwise suspend the program
        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"  
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount

#================================================ Value Iteration =========================================================
    # Value Iteration in RL : 
    # when the DIF between new and old values is smaller than the threshold, 
    # we see it as convergence, which means that the policy is the optimal
    def valueIteration(self, initialV, nIterations= np.inf, tolerance= 0.01):
        '''Value iteration procedure : 
        V <-- max_a R^a + gamma T^a V (Calculate the current state values of all states by 1 iteration)

        Inputs:
        initialV (Init. state values of each state) -- Initial value function: array of |S| entries 
        nIterations (limitation of total iterations used to find the optimal policy) -- limit on the # of iterations: scalar (default: infinity)
        tolerance (convergence condition) -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId (to see how many iterations it takes to converge the state values (i.e. find the optimal policy)) -- # of iterations performed: scalar
        epsilon (current DIF between old & new state values) -- ||V^n-V^n+1||_inf: scalar'''  

        V = initialV
        iterId = 0
        epsilon = np.inf  # init.
        while epsilon > tolerance:
            # iterate 1 times of the state values of all states in each Value Update process
            temp_v = np.zeros(shape= (self.nStates, self.nActions))
            iterId += 1
            for s in range(self.nStates):
                for a in range(self.nActions):
                    temp_v[s][a] = self.R[a][s] + np.sum(self.discount * self.T[a][s][np.nonzero(self.T[a][s])] * V[np.nonzero(self.T[a][s])])
            new_v = np.max(temp_v, axis= 1)
            epsilon = np.linalg.norm(new_v - V, np.inf)  # calculate || ||_inf
            V = new_v.copy()

        # while iterId < nIterations and epsilon > tolerance:
        #     Ta_V = np.matmul(self.T, V)
        #     gamma_Ta_V = self.discount * Ta_V
        #     all_possible_values = self.R + gamma_Ta_V
        #     policy = np.argmax(all_possible_values, axis=0)  # Choose the best actions for each state, policy means keep
        #     V_new = np.amax((all_possible_values), axis=0)  # Choose the best action values for each state
        #     # np.round/np.around does not work for 0.5 so not reducing to 2 decimal places
        #     V_diff = (V_new - V)
        #     V = V_new
        #     epsilon = np.linalg.norm(V_diff, np.inf)
        #     iterId = iterId + 1

        return [V, iterId, epsilon]
        

    def extractPolicy(self, V):
        '''Procedure to extract a policy from a value function : just return the action with the max. {R + state value}
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        temp_policy = np.zeros(shape= (self.nStates, self.nActions))
        policy = np.zeros(shape= self.nStates)
        for s in range(self.nStates):
            for a in range(self.nActions):
                temp_policy[s][a] = self.R[a][s] + self.discount * np.sum(self.T[a][s][np.nonzero(self.T[a][s])] * V[np.nonzero(self.T[a][s])])
        policy = np.argmax(temp_policy, axis= 1)
        return policy



#================================================= Policy Iteration ============================================================
    def evaluatePolicy(self, policy):
        '''Evaluate a policy by solving a system of linear equations : solve the BE completely (iterate until DIF between new and old < 0.01 (Convergence))
        V^pi = R^pi + gamma T^pi V^pi (if there are 2 possible next state, than use Weighted Sum (Weight is the prob.))

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        V = np.zeros(self.nStates)  # Assume InitalV = np.array([0, 0, 0, 0])
        new_v = np.zeros(self.nStates)
        tolerance = 0.1  # threshold of convergence
        epsilon = 5
        iter_no = 0
        while epsilon > tolerance and iter_no < 1000:
            iter_no += 1
            for s in range(self.nStates):
                new_v[s] = self.R[policy[s]][s] + np.sum(self.discount * self.T[policy[s]][s][(np.nonzero(self.T[policy[s]][s]))] * V[np.nonzero(self.T[policy[s]][s])])
            epsilon = np.max(new_v - V)
            V = new_v.copy()
        return V

        
    def policyIteration(self, initialPolicy, nIterations= np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).
        keep iterate unitl \pi_{t+1} = \pi_{t}

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by policy iteration: scalar'''

        iterId = 0
        policy = np.zeros(self.nStates)
        old_policy = initialPolicy
        
        while iterId < nIterations:
            iterId += 1
            V = self.evaluatePolicy(policy= old_policy)
            policy = self.extractPolicy(V)  # extract policy from V
            if np.array_equal(policy, old_policy): break
            old_policy = policy.copy()
        return [policy, V, iterId]

#=================================================== Truncated Policy Iteration =========================================================   
    def evaluatePolicyPartially(self, policy, initialV, nIterations= np.inf, tolerance= 0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        nEvalIterations (added by Wu) -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V = initialV
        new_v = np.zeros(self.nStates)
        epsilon = np.inf  # init.
        iterId = 0
        while iterId < nIterations and epsilon > tolerance:  # iterate "nEvalIterations" times in each PE
            iterId += 1
            for s in range(self.nStates):
                new_v[s] = self.R[policy[s]][s] + np.sum(self.discount * self.T[policy[s]][s][(np.nonzero(self.T[policy[s]][s]))] * V[np.nonzero(self.T[policy[s]][s])])
            epsilon = np.linalg.norm(new_v - V, np.inf)
            V = new_v.copy()
        
        return [V, iterId, epsilon]

    def modifiedPolicyIteration(self, initialPolicy, initialV, nEvalIterations= 5, nIterations= np.inf, tolerance= 0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)
        keep iterate until ||V^n-V^n+1||_inf < tolerance 

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        iterId = 0
        epsilon = np.inf
        old_policy = initialPolicy
        pre_v = initialV
        policy = np.zeros(self.nStates)
        V = np.zeros(self.nStates)
        while iterId < nIterations and epsilon > tolerance:
            iterId += 1
            [V, _, _] = self.evaluatePolicyPartially(policy= old_policy, initialV= pre_v, nIterations= nEvalIterations)
            policy = self.extractPolicy(V)  # extract policy from V
            epsilon = np.linalg.norm(V - pre_v, np.inf)
            old_policy = policy.copy()
            pre_v = V.copy()

        return [policy, V, iterId, epsilon]
        