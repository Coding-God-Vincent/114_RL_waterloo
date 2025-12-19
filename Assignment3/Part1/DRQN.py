import numpy as np
import utils_DRQN.envs, utils_DRQN.seed, utils_DRQN.buffers, utils_DRQN.torch
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings("ignore")

# Deep Recurrent Q Learning
# Slide 17
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module4.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = utils_DRQN.torch.TorchHelper()
DEVICE = t.device
OBS_N = 2               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.05       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 100   # episodes used to warm-up
TRAIN_EPOCHS = 5        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 2000         # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 50 # Target network update frequency

# Global variables
EPSILON = STARTING_EPSILON
Q = None

# Deep recurrent Q network
class DRQN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.obs_extractor = nn.Linear(in_features= OBS_N, out_features= HIDDEN, dtype= torch.float32)
        self.lstm = nn.LSTM(input_size= HIDDEN, hidden_size= HIDDEN, batch_first= True)
        self.Q_network = nn.Linear(in_features= HIDDEN, out_features= ACT_N)
    
    # Sequence Length : T, length of an episode
    # x : current observation with shape (batch_size, Sequence_Length, OBS_N)
    # hidden : initial hidden state (h_0, c_0), shape ((1, batch_size, HIDDEN), (1, batch_size, HIDDEN))
    # return q_values with shape (batch_size * Sequence_Length, ACT_N)
    def forward(self, x, hidden):
        # shape (batch_size, Sequence_Length, HIDDEN)
        extracted_obs = self.obs_extractor(x)
        # output: h0, h1, ..., h_(Sequence_Length), shape (batch_size, Sequence_Length, HIDDEN)
        # h_n, c_n: shape (1, batch_size, HIDDEN)
        output, (h_n, c_n) = self.lstm(extracted_obs, hidden)
        # shape (batch_size, Sequence_Length, ACT_N)
        Q_values = self.Q_network(output)
        return Q_values, (h_n, c_n)


# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
def create_everything(seed):
    utils_DRQN.seed.seed(seed)
    env = utils_DRQN.envs.TimeLimit(utils_DRQN.envs.PartiallyObservableCartPole(), 200)
    # env.seed(seed)
    test_env = utils_DRQN.envs.TimeLimit(utils_DRQN.envs.PartiallyObservableCartPole(), 200)
    # test_env.seed(seed)
    buf = utils_DRQN.buffers.ReplayBuffer(BUFSIZE)
    Q = DRQN().to(DEVICE)
    Qt = DRQN().to(DEVICE)
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

# Create epsilon-greedy policy
# TODO: Adjust this policy to handle hidden states?
# obs: np.array with shape (2)
# h, c: hidden state & cell state of the previous time, shape (1, 1, HIDDEN)
def policy(env, obs, h, c):

    global EPSILON, EPSILON_END, STEPS_MAX, Q
    obs = t.f(obs).reshape(1, 1, OBS_N)  # Convert to torch tensor, shape (1, 1, OBS_N)
    # Q_values of the current obs with shape (1, 1, ACT_N)
    with torch.no_grad():
        Q_value, (h_n, c_n) = Q(obs, (h, c))
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)  # pure int
    else:
        action = torch.argmax(Q_value, dim= 2).item()  # pure int
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action, (h_n, c_n)


# Update networks
def update_networks(epi, buf, Q, Qt, OPT):
    losses = []
    loss = 0.
    # S.shape = (batch_size, Sequence_Length, OBS_N)
    # A.shape = (batch_size, Sequence_Length)
    # R.shape = (batch_size, Sequence_Length)
    # S2.shape = (batch_size, Sequence_Length, OBS_N)
    # D.shape = (batch_size, Sequence_Length)
    S_, A_, R_, S2_, D_ = buf.sample(n= MINIBATCH_SIZE, t= t)
    
    # Sequence_Length = 10

    # # update by 1 episode at a time
    for i in range(MINIBATCH_SIZE):
    #     if (len(S_[i]) > Sequence_Length):
    #         T_start = random.randint(0, len(S_[i])-Sequence_Length)
    #         T_end = T_start + Sequence_Length
    #     else: 
    #         T_start = 0
    #         T_end = len(S_[i])
    #     # shape (1, Sequence_Length, OBS_N)
    #     S = torch.tensor(S_[i][T_start : T_end], dtype= torch.float32, device= DEVICE).unsqueeze(dim= 0)
    #     # shape (1, Sequence_Length)
    #     A = torch.tensor(A_[i][T_start : T_end], dtype= torch.long, device= DEVICE).unsqueeze(dim= 0)
    #     # shape (1, Sequence_Length)
    #     R = torch.tensor(R_[i][T_start : T_end], dtype= torch.float32, device= DEVICE).unsqueeze(dim= 0)
    #     # shape (1, Sequence_Length, OBS_N)
    #     S2 = torch.tensor(S2_[i][T_start : T_end], dtype= torch.float32, device= DEVICE).unsqueeze(dim= 0)
    #     # shape (1, Sequence_Length)
    #     D = torch.tensor(D_[i][T_start : T_end], dtype= torch.int32, device= DEVICE).unsqueeze(dim= 0)
        
        S = torch.tensor(S_[i], dtype= torch.float32, device= DEVICE).unsqueeze(dim= 0)
        # shape (1, Sequence_Length)
        A = torch.tensor(A_[i], dtype= torch.long, device= DEVICE).unsqueeze(dim= 0)
        # shape (1, Sequence_Length)
        R = torch.tensor(R_[i], dtype= torch.float32, device= DEVICE).unsqueeze(dim= 0)
        # shape (1, Sequence_Length+1 (0~T), OBS_N)
        # 若不加上 s0，這樣時間資訊不完整，進而導致訓練失敗
        # 後面 Qt 輸出再把 s_0 的剪掉
        S2 = torch.tensor(S_[i] + [S2_[i][-1]], dtype= torch.float32, device= DEVICE).unsqueeze(dim= 0)
        # shape (1, Sequence_Length)
        D = torch.tensor(D_[i], dtype= torch.int32, device= DEVICE).unsqueeze(dim= 0)

        batch_size, Sequence_Length, OBS_N = S.shape
        h_init = torch.zeros(size= (1, batch_size, HIDDEN), dtype= torch.float32, device= DEVICE)
        c_init = torch.zeros(size= (1, batch_size, HIDDEN), dtype= torch.float32, device= DEVICE)
    
        # shape = (batch_size, Sequence_Length, ACT_N)
        Q_values, (h_n, c_n) = Q(S, (h_init, c_init))
        # shape (batch_size, Sequence_Length, 1) with dtype (torch.long or torch.int64)
        indices = A.unsqueeze(dim= 2)
        # shape (batch_size, Sequence_Length)
        Q_values = Q_values.gather(dim= 2, index= indices).squeeze(dim= 2)
    
        # shape = (batch_size, Sequence_Length+1, ACT_N)
        with torch.no_grad():
            Q2_values, (h_n, c_n) = Qt(S2, (h_init, c_init))
        # shape (batch_size, Sequence_Length)
        Q2_values = torch.max(Q2_values[:, 1:, :], dim= 2).values
        
        # shape (batch_size, Sequence_Length)
        targetQ_values = R + GAMMA * Q2_values * (1-D)
        
        # shape ()
        loss = torch.nn.MSELoss()(targetQ_values.detach(), Q_values)  
        losses.append(loss)

    total_loss = torch.stack(losses).mean()
    # update network
    OPT.zero_grad()
    total_loss.backward()
    OPT.step()

    # Update target network
    if epi%TARGET_NETWORK_UPDATE_FREQ==0:
        Qt.load_state_dict(Q.state_dict())

    return loss


# Play episodes
# Training function
def train(seed):

    global EPSILON, Q
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, Q, Qt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils_DRQN.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Q, Qt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils_DRQN.envs.play_episode(test_env, policy, render = False)
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
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'drqn')
    plt.legend(loc='best')
    plt.savefig('/home/super_trumpet/NCKU/Homework/114-2-DRL/Assignment3/Part1/DRQN_result')