import numpy as np
import cv2
from collections import deque
from pandas import Series
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym

def preprocess_state(state):
    state = cv2.resize(state, (84, 84))
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(state, dtype=np.float32) / 255

class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)

    def reset(self):
        obs, info = self.env.reset()
        obs = preprocess_state(obs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = preprocess_state(obs)
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.array(self.frames)

def ewma(x, span=100):
    return Series(x).ewm(span=span).mean()

def plot_training_progress(rewards, actor_losses, critic_losses, entropies, iterations, save=False):
    plt.figure(figsize=(20, 16))
    
    # Plot episode rewards and average rewards
    plt.subplot(3, 2, 1)
    plt.title('Episode Rewards and Learning Curve')
    plt.plot(range(len(rewards)), rewards, alpha=0.3, label='Episode Rewards')
    window_size = 100
    average_rewards = [np.mean(rewards[max(0, i-window_size):i]) for i in range(1, len(rewards)+1)]
    plt.plot(range(len(average_rewards)), average_rewards, label='Average Reward (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Plot actor loss
    plt.subplot(3, 2, 2)
    plt.title('Actor Loss')
    plt.plot(iterations, actor_losses, label='Actor Loss')
    plt.plot(iterations, ewma(np.array(actor_losses), span=1000), label='Actor Loss EWMA@1000')
    plt.xlabel('Total Steps')
    plt.ylabel('Loss')
    plt.legend()

    # Plot critic loss
    plt.subplot(3, 2, 3)
    plt.title('Critic Loss')
    plt.plot(iterations, critic_losses, label='Critic Loss')
    plt.plot(iterations, ewma(np.array(critic_losses), span=1000), label='Critic Loss EWMA@1000')
    plt.xlabel('Total Steps')
    plt.ylabel('Loss')
    plt.legend()

    # Plot entropy
    plt.subplot(3, 2, 4)
    plt.title('Policy Entropy')
    plt.plot(iterations, entropies, label='Entropy', alpha=0.3)
    plt.plot(iterations, ewma(np.array(entropies), span=1000), label='Entropy EWMA@1000')
    plt.xlabel('Total Steps')
    plt.ylabel('Entropy')
    plt.legend()

    plt.tight_layout()
    
    if save:
        # Save the plot
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        plt.savefig('outputs/PPO_Continuous_training.png')
        plt.close()
    else:
        plt.show()

def evaluate_agent(agent, env, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)

def record_video(env, agent, video_length=500, filename="trained_ppo_continuous_agent_video.mp4"):
    frames = []
    state, _ = env.reset()
    episode_reward = 0
    
    for _ in range(video_length):
        frame = env.render()
        frames.append(frame)
        
        action, _ = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    height, width, _ = frames[0].shape
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    video = cv2.VideoWriter(f'outputs/{filename}', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    
    for frame in frames:
        video.write(frame)
    
    video.release()
    print(f"Video saved as outputs/{filename}")
    return episode_reward