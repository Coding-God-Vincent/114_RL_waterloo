import torch
import numpy as np
import collections
import random
# a = torch.tensor([[1, 2]])
# print(a.shape, a.reshape(1, 1, 2).shape, a.reshape(1, 1, 2))
# print(torch.argmax(a, dim= 1).item())  # shape (1)
# print(np.random.randint(2))  # shape ()

# experience = []
# s1 = np.array([1, 2])
# a1 = 1
# r1 = 1
# s2 = np.array([3, 4])
# a2 = 2
# r2 = 5
# exp1 = (s1, a1, r1)
# exp2 = (s2, a2, r2)
# experience.append(exp1)
# experience.append(exp2)
# print(experience)  # experience = [exp1, exp2]

# for e in experience:
#     s, a, r = e
#     print(s)
#     print(a)

# buf = collections.deque(maxlen= 5)
# exp = [(1, 2, 3), (2, 3, 4)]
# exp2 = [(12, 13, 14), (15, 16, 17)]
# buf.append(exp)
# buf.append(exp2)
# # buf = [[(1, 2, 3), (2, 3, 4)], [(12, 13, 14), (15, 16, 17)]]
# minibatch = random.sample(buf, 2)
# print(minibatch)
# # minibatch = [[(1, 2, 3), (2, 3, 4)], [(12, 13, 14), (15, 16, 17)]]
# for trajectory in minibatch:
#     # print(trajectory)  # [(1, 2, 3), (2, 3, 4)]
#     # break
#     for exp in trajectory:
#         print(exp)  # (1, 2, 3) then (2, 3, 4)

# a = [([1, 2], 2, 3), ([3, 4], 5, 6)]
# S = []; A = []
# for a_ in a:
#     s, k, b = a_
#     print(f"s = {s}, k = {k}, b = {b}")
#     S += [s]; A += [k]
# print(A)  # S = [[1, 2], [3, 4]]

# a = torch.tensor([[[1, 2]]])  # a.shape (1, 1, 1)
# print(type(torch.argmax(a, dim= 2).item()))


# shape (2, 3, 3)
input = torch.tensor([
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],  # batch 0
    [[20, 21, 22], [23, 24, 25], [26, 27, 28]]   # batch 1
]) 

# shape(2, 3, 1)
# indices = torch.tensor([
#     [[0], [1], [2]],
#     [[2], [1], [0]]
# ], dtype= torch.long)
# shape (2, 3)
# indices = torch.tensor([
#     [0, 1, 2],
#     [2, 1, 0]
# ], dtype= torch.long)
# indices = indices.view(-1, 1)
# print(indices, indices.dtype)  # shape (2, 3, 1)

# output = input.gather(dim= 2, index= indices)

# # output: shape (2, 3, 1)        
# tensor([[[10],
#          [14],
#          [18]],

#         [[22],
#          [24],
#          [26]]])

# print(output)

# print(input.argmax(dim= 2))  # shape (2, 3)

# qvalues = torch.tensor([[[1], [2], [3]]], dtype= torch.float32)  # shape (3, 1, 1)
# target_qvalues = torch.tensor([[[3], [4], [5]]], dtype= torch.float32)  # shape (3, 1, 1)
# print(torch.nn.MSELoss()(qvalues, target_qvalues))

# a = [1, 2, 3, 4, 5, 6]
# b = [1, 2, 3]
# final = []
# final.append(a[0:len(b)])
# final.append(b[0:len(b)])
# print(final)

# a = torch.tensor([1, 2, 3, 4, 5])
# print(a[1:])

# support = np.linspace(start= 0, stop= 200, num= 51)
# print(support)

# ATOMS = 51              # Number of atoms for distributional network
# ZRANGE = [0, 200] 
# print((ZRANGE[1] - ZRANGE[0]) / 50)

# a = torch.tensor([1, 2, 3], dtype= torch.float32)
# b = torch.tensor([4, 5, 6], dtype= torch.float32)
# b_ = torch.nn.functional.softmax(b)
# CEL = torch.nn.functional.cross_entropy(a, b_)
# print(CEL)

# a = torch.tensor([[1, 2, 3, 4, 5, 6]])  # shape (1, 6)
# a = a.reshape((1, 2, 3))  # shape (1, 2, 3)
# k = torch.tensor([[[1, 2, 3],  # shape (1, 2, 3)
#          [4, 5, 6]]])
# print(k[0], k[0].shape)
# k = torch.ones((1, 2, 51)).numpy()  # np.array with (1, 2, 51)
# torch.max(k, dim= 2).values  # return shape (1, 2)
# print(torch.max(a), torch.max(k, dim= 2), )
# expectations_each_action = [k_ * k[0, a, :] for a in range(2)] 
# print(expectations_each_action, expectations_each_action.shape)

# SUPPORT = np.linspace(start= ZRANGE[0], stop= ZRANGE[1], num= ATOMS)
# expectations_each_action = [SUPPORT @ k[0, a, :] for a in range(2)]  # shape (2)
# action = int(np.argmax(expectations_each_action))

# a = []
# a += [1]; a += [2]


# print(torch.tensor(a))  # shape (2)

# a = [1, 2, 3]
# print(np.argmax(a, axis= 0))

# print(np.zeros(51, dtype= np.float32).shape)

# a = torch.tensor([1, 2, 3])  # shape (3)
# a = a.unsqueeze(dim= -1).unsqueeze(dim= -1)  # shape (3, 1, 1)
# a = a.expand((3, 1, 5))  # shape (3, 1, 5)
# print(a, a.shape)

# (batch_size, ACT_N, ATOMS)
# dist_batch = torch.tensor([  # shape (2, 2, 3)
#     [[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]
# ])

# (batch_size, ACT_N, ATOMS)
# dist_batch = torch.tensor([  # shape (2, 1, 3)
#     [[1, 2, 3]], [[7, 8, 9]]
# ])
# (2, 1, 1)
# reward = torch.tensor([[[10.5]], [[20.7]]])
# print(dist_batch + reward)  # shape (2, 1, 3)
# L = torch.floor(dist_batch + reward)
# U = torch.ceil(dist_batch + reward)
# print(L, U)

# print(dist_batch[1, 1, :])  # shape (3)
# print(dist_batch[0])  # shape (2, 3)

# indices = torch.tensor([  # shape (2, 1, 3)
#     [[1, 1, 1]], [[1, 1, 1]]
# ], dtype= torch.int64)

# outcome = torch.gather(dist_batch, dim= 1, index= indices)
# print(outcome, outcome.shape)  # shape (2, 1, 3)


# dist_batch = torch.tensor([  # shape (2, 2, 3)
#     [[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]
# ], dtype= torch.float32)

# print(torch.nn.functional.softmax(dist_batch, dim= 2))  # shape (2, 2, 3)

# a = torch.tensor([  # shape (2, 1, 3)
#     [[1, 2, 3]], [[7, 8, 9]]
# ], dtype= torch.float32)
# # print(dist_batch * a)

# index = torch.tensor([
#     [[0, 0, 2]], [[1, 0, 2]]
# ], dtype= torch.long)  # shape (2, 1, 3)

# probs = torch.tensor([  # shape (2, 1, 3)
#     [[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]
# ], dtype= torch.float32)

# for i in range(2):
#     a[i, 0, index[i, 0, :]] += probs[i, 0, :]
# print(a)

# print(dist_batch[:, 0, index[:, 0, :]])

# support = np.array([1, 2, 3])
# support_set = torch.from_numpy(support)  # shape (3)
# support_set = support_set.reshape(1, 1, 3)
# support_set = support_set.expand(3, 1, 3)
# print(support_set, support_set.shape)

# a = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype= torch.float32)  # shape (3, 2)
# index = torch.tensor([0, 1, 0], dtype= torch.long)  # 對 0th、1th、0th row
# source = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.]], dtype= torch.float32)
# a.index_add_(dim= 0, index= index, source= source)
# print(a) # a = torch.tensor([[1.6, 2.2], [2.3, 3.4], [3., 4.]])

# a = torch.tensor([[1, 5], [2, 4], [3, 6]])  # shape (3, 2)
# a[0] = torch.tensor([5, 6])
# print(a[0, :])  # shape (2)

# a = torch.tensor([1, 2, 3])  # shape (3)
# index = torch.tensor([1, 1, 2], dtype= torch.long)
# source = torch.tensor([5, 6, 7])
# a.index_add_(dim= 0, index= index, source= source)
# print(a)

# a = torch.tensor([1, 2, 3])
# index = torch.tensor([0, 1, 2], dtype= torch.long)
# b = torch.tensor([10, 11, 12])
# b[:] = a[index]
# print(b)

a = torch.tensor([0.2, 0.3], dtype= torch.float32)
b = torch.tensor([0.5, 0.3], dtype= torch.float32)
print(torch.where(a == b)[0])