import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        self.memory = deque(maxlen=2000)  # Experience replay memory
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size).to(device)  # Initialize the DQN model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer
        self.criterion = nn.MSELoss()  # Mean Squared Error loss

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: random action
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            act_values = self.model(state)  # Exploit: use the model to predict the best action
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        # Train the model using a batch of experiences from memory
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor([reward]).to(device)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state)))
            target_f = self.model(state)
            target_f = target_f.clone().detach()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Load a saved model
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        # Save the model
        torch.save(self.model.state_dict(), name)

if __name__ == "__main__":
    # Main training loop
    env = AdaptiveCruiseControlEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)  # Agent takes action
            next_state, reward, done, _ = env.step(action)  # Environment responds to action
            reward = reward if not done else -10  # Adjust reward if done (collision)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)  # Remember the experience
            state = next_state  # Move to the next state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # Train the agent with the experience replay

        if e % 50 == 0:
            agent.save(f"acc-dqn-{e}.pth")  # Save the model every 50 episodes
