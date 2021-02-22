import torch
import torch.nn as nn
import torch.optim as optim

#class DQN(nn.Module):
#
#    def __init__(self, env_actions):
#        super(DQN, self).__init__()
#        # input: an observation vector
#        self.conv1 = nn.Conv2d(1, 16, kernel_size=, stride=)
#        self.conv2 = nn.Conv2d(16, 32, kernel_size=, stride=)
#        self.fc    = nn.Linear(, )
#        self.out   = nn.Linear(, env_actions)
#
#    def forward(self, x):
#        x = torch.nn.functional.relu(self.conv1(x))
#        x = torch.nn.functional.relu(self.conv2(x))
#        x = self.fc(x)
#
#        return self.out(x)



# another network class that takes as input a "frame" ???
