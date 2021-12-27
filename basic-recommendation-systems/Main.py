import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', header=None, encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', engine='python', header=None, encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python', header=None, encoding='latin-1')

training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# Convert Training and Test sets into a matrix
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert the ratings into Binary (Liked:1, Unliked:0)
training_set[training_set == 0] = -1
test_set[test_set == 0] = -1
training_set[training_set >= 3] = 1
training_set[training_set == 2] = 0
training_set[training_set == 1] = 0
test_set[test_set >= 3] = 1
test_set[test_set == 2] = 0
test_set[test_set == 1] = 0


# Creating the architecture of the NN
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)  # Bias for hidden nodes
        self.b = torch.randn(1, nv)  # Bias for visible nodes and bias for probability p_v_given_h

    def sample_h(self,
                 x):  # Activates hidden nodes according to a probability P(h|v); x is the visible neurons in the probability p(h|v)
        wx = torch.mm(x,
                      self.W.t())  # Sigmoid function = wx + bias; p_h_given_v is the probability that the hidden node is activated
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):  # ph0 is the probability that at the 1st iteration, hidden node =1 given v0
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)  # To keep the tensor two dimensional
        self.a += torch.sum((ph0 - phk), 0)


nv = len(training_set[0])
nh = 300
batch_size = 25
rbm = RBM(nv, nh)

# training the RBM
nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size]  # vk = Inputs that will change over k-iterations
        v0 = training_set[id_user:id_user + batch_size]  # Because at the beginning, the input is the same as the target
        ph0, _ = rbm.sample_h(v0)

        for k in range(10):
            _, hk = rbm.sample_h(vk)  # Hidden nodes obtained at the kth step in Contrastive divergence
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch : ' + str(epoch) + ' loss: ' + str(train_loss / s))

# Testing the RBM

test_loss = 0
s = 0.
v_all = np.empty(shape=[0, nb_movies])
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1]  # vk = Inputs that will change over k-iterations
    v_numpy = v.numpy()
    vt = test_set[id_user:id_user + 1]  # Because at the beginning, the input is the same as the target
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)  # Hidden nodes obtained at the kth step in Contrastive divergence
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
        v_all = np.vstack([v_all, v_numpy])
print('test loss: ' + str(test_loss / s))
