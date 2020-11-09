# Author: Sida
# Description:
# This is an implementation of using a random network as policy


import time
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F

from policy import Policy


class Policy_Random_Network(Policy):
    def __init__(self, dim_obs=3, dim_action=2, seed=0):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles = None
        if seed is None:
            self.seed = int(time.time())
        else:
            self.seed = seed
        np.random.seed(seed)  # generate a new system every time
        torch.manual_seed(seed)
        self.net = Net(dim_obs, dim_action, 16)
        # Path("data_random_network").mkdir(parents=True, exist_ok=True)
        # with open(f"data_random_network/{seed}.pickle", "wb") as f:
        #     pickle.dump( (self.weights, self.bias), f)
        print(f"Random network with seed {seed}")

    def get_action(self, obs):
        """
        obs:
            refer to world.get_obs().
        """
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0]
        obs = torch.Tensor(obs)
        ret = self.net(obs)
        return ret


class Net(nn.Module):
    def __init__(self, dim_obs, dim_action, num_hidden_nodes):
        super().__init__()
        self.fc1 = nn.Linear(dim_obs, num_hidden_nodes)
        self.fc2 = nn.Linear(num_hidden_nodes, dim_action)

    def forward(self, x):
        with no_grad():
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x.numpy()
