import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from src.model import ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, num_actions, lr=3e-4, gamma=0.99, gae_lambda=0.95, ppo_epochs=10, clip_epsilon=0.2):
        self.policy = ActorCritic(num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std, _ = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], action_log_prob.item()
    
    def update(self, states, actions, rewards, next_states, dones, log_probs):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        old_log_probs = torch.FloatTensor(log_probs).to(device)

        # Compute GAE
        with torch.no_grad():
            _, _, values = self.policy(states)
            _, _, next_values = self.policy(next_states)

        advantages = torch.zeros_like(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            mean, std, values = self.policy(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            entropy = dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()