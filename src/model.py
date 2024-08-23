import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(3136, 512)
        self.actor_mean = nn.Linear(512, num_actions)
        self.actor_log_std = nn.Parameter(torch.zeros(1, num_actions))
        self.critic = nn.Linear(512, 1)
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        action_mean = torch.tanh(self.actor_mean(x))
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        value = self.critic(x)
        
        return action_mean, action_std, value