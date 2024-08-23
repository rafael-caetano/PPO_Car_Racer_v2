import gymnasium as gym
from tqdm import tqdm
from IPython.display import clear_output
import torch
import os

from src.utils import FrameStack, plot_training_progress
from src.ppo import PPO

def train_ppo(num_episodes=2000, max_steps=1000, update_interval=2048, plot_interval=10):
    env = gym.make('CarRacing-v2', render_mode="rgb_array")
    env = FrameStack(env)
    ppo_agent = PPO(num_actions=env.action_space.shape[0])

    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
    episode_rewards = []
    actor_losses, critic_losses, entropies = [], [], []
    iterations = []

    total_steps = 0
    update_count = 0

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action, log_prob = ppo_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(log_prob)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if len(states) >= update_interval:
                actor_loss, critic_loss, entropy = ppo_agent.update(states, actions, rewards, next_states, dones, log_probs)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                entropies.append(entropy)
                states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
                update_count += 1
                iterations.append(total_steps)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % plot_interval == 0 and update_count > 0:
            clear_output(True)
            plot_training_progress(episode_rewards, actor_losses, critic_losses, entropies, iterations)

    env.close()

    # Save the trained model
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    torch.save(ppo_agent.policy.state_dict(), 'outputs/ppo_continuous_model.pth')

    return ppo_agent, episode_rewards, actor_losses, critic_losses, entropies, iterations