import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
1) CNN Network architecture used in the Nature paper
   The input to the network consists of an image with shape (1, 84, 84)
2) Linear Mapping Network
   The input to the network consists of the observation/state vector
""" 
class CNN(nn.Module):

    def __init__(self, env_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc    = nn.Linear(3136, 512) # 64 x 7 x 7
        self.out   = nn.Linear(512, env_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv_to_fc(x)
        x = F.relu(self.fc(x))

        return self.out(x)

    def conv_to_fc(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return x.view(-1, num_features)


class LinearMapNet(nn.Module):
    def __init__(self, input_shape, env_actions):
        super(LinearMapNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, env_actions)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return self.out(x)
