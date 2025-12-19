import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
# SEEDS = [1]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 500          # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Suggested constants
ATOMS = 51              # Number of atoms for distributional network
ZRANGE = [0, 200]       # Range for Z projection

# key elements in C51
SUPPORT = np.linspace(start= ZRANGE[0], stop= ZRANGE[1], num= ATOMS)  # support of C51, np.array with shape (51)
INTERVAL = (ZRANGE[1] - ZRANGE[0]) / (ATOMS - 1)  # interval between items in the support 

# Global variables
EPSILON = STARTING_EPSILON
Z = None

# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    # env.seed(seed)
    test_env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    # test_env.seed(seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    #  input shape (batch_size, OBS_N)
    #  output shape (batch_size, ACT_N*ATOMS)
    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS),
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Z, Zt, OPT

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, EPSILON_END, STEPS_MAX, Z
    obs = t.f(obs).view(-1, OBS_N)  # Convert from list to torch tensor, shape (1, OBS_N)
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        # dist : Logits of each Q(s, a), shape (1, ACT_N*ATOMS)
        # then change to shape (1, ACT_N, ATOMS)
        # then do softmax
        with torch.no_grad():
            dist = torch.nn.functional.softmax(Z(obs).reshape(1, ACT_N, ATOMS), dim= 2).cpu().numpy()
        # SUPPORT : np.array with shape (51)
        expectations_each_action = [SUPPORT @ dist[0, a, :] for a in range(ACT_N)]  # [np.int64, np.int64]
        action = int(np.argmax(expectations_each_action, axis= 0))  # 取出期望值較大的動作

    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# Update networks
def update_networks(epi, buf, Z, Zt, OPT):
    
    # device = DEVICE
    # S : shape (batch_size, OBS_N), dtype= torch.float32
    # A : shape (batch_size), dtype= torch.long (torch.int64)
    # R : shape (batch_size), dtype= tortch.float32
    # S2 : shape (batch_size, OBS_N), dtype= torch.float32
    # D : shape (batch_size), dtype= torch.int32
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    
    # Z(S) logits, shape (batch_size, ACT_N, ATOMS)
    dist_batch_logits = Z(S).reshape((MINIBATCH_SIZE, ACT_N, ATOMS))
    # Z(S), shape (batch_size, ACT_N, ATOMS)
    dist_batch = torch.nn.functional.softmax(dist_batch_logits, dim= 2)
    
    # extract Z(S, A), shape (batch_size, 1, ATOMS)
    A = A.unsqueeze(dim= -1).unsqueeze(dim= -1)  # shape (batch_size, 1, 1)
    A = A.expand((MINIBATCH_SIZE, 1, ATOMS))  # shape (batch_size, 1, ATOMS)
    dist_batch_action = torch.gather(dist_batch, dim= 1, index= A)  # shape (batch_size, 1, ATOMS) # gather will preserve the grad.
    dist_batch_logits_action = torch.gather(dist_batch_logits, dim= 1, index= A)  # shape (batch_size, 1, ATOMS)

    # Zt(S2) : shape (batch_size, ACT_N, ATOMS)
    with torch.no_grad():
        # shape (batch_size, ACT_N, ATOMS)
        dist_batch2 = torch.nn.functional.softmax(Zt(S2).reshape((MINIBATCH_SIZE, ACT_N, ATOMS)), dim= 2)
    
    A2 = []
    # Gets A2
    for b in range(MINIBATCH_SIZE):
        # SUPPORT : np.array with shape (51)
        # [np.int64, np.int64]
        expectations_each_action = [SUPPORT @ dist_batch2[b, a, :].cpu().numpy() for a in range(ACT_N)]
        a2 = int(np.argmax(expectations_each_action, axis= 0))
        A2.append(a2)  # length = batch_size
    A2 = torch.tensor(A2, dtype= torch.long, device= DEVICE)  # shape (batch_size)
    A2 = A2.unsqueeze(dim= -1).unsqueeze(dim= -1).expand(MINIBATCH_SIZE, 1, ATOMS)  # shape (batch_size, 1, ATOMS)
    
    # Z(S2, A2) : we compute target_dist based on Z(S2, A2) (it's a prob. dist.), shape (batch_size, 1, ATOMS)
    dist_batch2_action = torch.gather(dist_batch2, dim= 1, index= A2)
    
    # create target_dist:
    #    update each atoms (zi) in Zt(s2, a2) by BOE (generate a new return which maynot be different from atoms) -> new_zi
    #    put the probs of zi in the new position (find the position based on new_zi)
    # initialize target_dist, shape (batch_size, 1, ATOMS)
    target_dist_batch_action = torch.zeros(size= (MINIBATCH_SIZE, 1, ATOMS), dtype= torch.float32, device= DEVICE)
    
    # in order to compute with tensor, create a SUPPORT in tensor
    SUPPORT_tensor = torch.from_numpy(SUPPORT).to(dtype= torch.float32, device= DEVICE)
    for b in range(MINIBATCH_SIZE):
        new_zi = torch.clamp((R[b] + (1 - D[b]) * GAMMA * SUPPORT_tensor), min= ZRANGE[0], max= ZRANGE[1])  # shape (ATOMS)
        new_zi_index = (new_zi - ZRANGE[0]) / INTERVAL  # shape (ATOMS)
        # 若 lower_index 為 50，這樣 upper_index 會超過，所以限制 lower_index 為 49
        lower_index = torch.floor(new_zi_index).to(dtype= torch.long)  # shape (ATOMS)
        lower_index = torch.clamp(lower_index, min= 0, max= ATOMS - 2)
        upper_index = lower_index + 1
        
        upper_prob = dist_batch2_action[b, 0, :] * (new_zi_index - lower_index)  # shape (ATOMS)
        lower_prob = dist_batch2_action[b, 0, :] * (upper_index - new_zi_index)# shape (ATOMS)
        # target_dist_batch_action[b, 0, :] : shape (ATOMS)
        # tensor.index_add_ : inplace 的對 index 位置的數值壘加 (若直接用 += 的話不會壘加)
        target_dist_batch_action[b, 0, :].index_add_(dim= 0, index= upper_index, source= upper_prob)
        target_dist_batch_action[b, 0, :].index_add_(dim= 0, index= lower_index, source= lower_prob)
        
    # Compute loss for the current data (預設會幫我們算該 batch 的 mean)
    # 比較的東西 (機率 & logit) 一定要放在 dim= 1，沒法改
    # input (pred) : logit，CE 內部會自己算 logSoftmax、target : prob. dist
    loss = torch.nn.functional.cross_entropy(
        input= dist_batch_logits_action.squeeze(dim= 1), 
        target= target_dist_batch_action.squeeze(dim= 1).detach()
    )
    
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # Update target network
    if epi%TARGET_NETWORK_UPDATE_FREQ==0:
        Zt.load_state_dict(Z.state_dict())

    return loss


# Play episodes
# Training function
def train(seed):

    global EPSILON, Z
    print("Seed=%d" % seed)

    # Create environment, buffer, Z, Z target, optimizer
    env, test_env, buf, Z, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Z, Zt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'c51')
    plt.legend(loc='best')
    plt.savefig('114-2-DRL/Assignment3/Part2/C51_result')