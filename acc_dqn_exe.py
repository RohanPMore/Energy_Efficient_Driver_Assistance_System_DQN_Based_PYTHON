import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gym
from gym import spaces

# Define the custom environment for Adaptive Cruise Control
class AdaptiveCruiseControlEnv(gym.Env):
    def __init__(self):
        super(AdaptiveCruiseControlEnv, self).__init__()
        self.max_speed = 30  # Max speed of the vehicle
        self.min_distance = 5  # Minimum safe distance to the lead car
        self.max_distance = 100  # Max distance to the lead car
        self.lead_car_speed = 15  # Speed of the lead car (fixed or variable)
        self.time_step = 1  # Time step in seconds
        self.action_space = spaces.Discrete(3)  # Actions: 0: decelerate, 1: maintain, 2: accelerate
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.max_speed, self.max_distance]), dtype=np.float32)
        self.reset()

    def reset(self):
        # Initialize the environment state
        self.speed = np.random.uniform(0, self.max_speed)  # Random initial speed
        self.distance_to_lead_car = np.random.uniform(self.min_distance, self.max_distance)  # Random initial distance
        return np.array([self.speed, self.distance_to_lead_car], dtype=np.float32)

    def step(self, action):
        # Update the environment state based on the action taken
        if action == 0:  # Decelerate
            self.speed = max(0, self.speed - 1)
        elif action == 2:  # Accelerate
            self.speed = min(self.max_speed, self.speed + 1)

        # Update the distance to the lead car based on the relative speeds and time step
        delta_distance = (self.lead_car_speed - self.speed) * self.time_step
        self.distance_to_lead_car = max(0, self.distance_to_lead_car + delta_distance)

        # Reward logic: penalize if too close to the lead car, otherwise reward
        reward = -1 if self.distance_to_lead_car < self.min_distance else 1

        # Check if the episode is done (collision with the lead car)
        done = self.distance_to_lead_car == 0
        state = np.array([self.speed, self.distance_to_lead_car], dtype=np.float32)
        return state, reward, done, {}

    def render(self, mode='human'):
        # Print the current state
        print(f'Speed: {self.speed}, Distance to lead car: {self.distance_to_lead_car}, Lead Car Speed: {self.lead_car_speed}')

# Define the Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size).to(device)  # Initialize the DQN model

    def load(self, name):
        # Load a saved model
        self.model.load_state_dict(torch.load(name))
        self.model.eval()  # Set the model to evaluation mode

    def evaluate_performance(self, env, episodes=10):
        episode_rewards = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = self.model(state).argmax(1).item()
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        return episode_rewards

if __name__ == "__main__":
    # Load the trained DQN agent
    env = AdaptiveCruiseControlEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("acc-dqn-950.pth")  # Load the model weights from the file

    # Evaluate the agent's performance over multiple episodes
    episode_rewards = agent.evaluate_performance(env, episodes=10)

    # Plot the rewards
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance')
    plt.grid(True)
    plt.show()
