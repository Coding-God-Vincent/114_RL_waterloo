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

print(random.randint(a= 0, b= 20))
