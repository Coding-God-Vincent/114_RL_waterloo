import collections
import numpy as np
import random
import torch

# Replay buffer
# TODO: Adjust this replay buffer to handle hidden states?
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N):
        self.buf = collections.deque(maxlen = N)
    
    # add: add an experience of an episode. ex: [(s1, a1, r1, s21, d1), (s2, a2, r2, s22, d2), ...]
    def add(self, exp):
        self.buf.append(exp)
    
    # sample: return minibatch of size n
    def sample(self, n, t):
        # minibatch = [trajectory1, trajectory2, ..., trajectoryT]
        minibatch = random.sample(self.buf, n)
        S, A, R, S2, D = [], [], [], [], []

        for trajectory in minibatch:  # trajectory = [exp1, exp2, ..., expT]
            S_, A_, R_, S2_, D_ = [], [], [], [], []
            for exp in trajectory:  # exp = (s, a, r, s2, done)
                s, a, r, s2, d = exp  # non-tensor elements
                # S_ = [s1, s2, ..., sT]
                S_.append(s); A_.append(a); R_.append(r), S2_.append(s2), D_.append(d)
            
            S.append(S_); A.append(A_); R.append(R_); S2.append(S2_); D.append(D_)
            
        # convert from list to tensor
        # return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)
        return S, A, R, S2, D
        
        # t.f(S).shape = (batch_size, Sequence_Length, OBS_N)
        # t.l(A).shape = (batch_size, Sequence_Length)
        # t.f(R).shape = (batch_size, Sequence_Length)
        # t.f(S2).shape = (batch_size, Sequence_Length, OBS_N)
        # t.i(D).shape = (batch_size, Sequence_Length)

