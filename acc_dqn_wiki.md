# Wiki Documentation for Adaptive Cruise Control with DQN

## Overview

This project implements an Adaptive Cruise Control (ACC) system using Deep Q-Learning (DQN). The ACC system is simulated in a custom OpenAI Gym environment, where the goal is to maintain a safe distance from the lead car by controlling the speed of the vehicle.

## Contents

- [Introduction](#introduction)
- [Environment](#environment)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [State and Reward](#state-and-reward)
- [Deep Q-Network (DQN)](#deep-q-network-dqn)
  - [Model Architecture](#model-architecture)
  - [Agent](#agent)
- [Training Loop](#training-loop)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction

The Adaptive Cruise Control system aims to automate speed control to maintain a safe distance from a lead vehicle. The system is trained using a Deep Q-Network (DQN) to learn the optimal policy for speed control.

## Environment

The custom Gym environment `AdaptiveCruiseControlEnv` simulates the ACC system.

### Observation Space

The observation space is a 2D continuous space represented by:
- `speed`: The current speed of the vehicle (0 to 30 units).
- `distance_to_lead_car`: The distance to the lead car (5 to 100 units).

### Action Space

The action space is a discrete space with three actions:
- `0`: Decelerate
- `1`: Maintain speed
- `2`: Accelerate

### State and Reward

- **State**: A state is represented as `[speed, distance_to_lead_car]`.
- **Reward**: A reward of `1` is given for maintaining a safe distance (greater than the minimum safe distance). A penalty of `-1` is given if the distance is less than the minimum safe distance. If a collision occurs (distance becomes 0), an additional penalty of `-10` is applied.

## Deep Q-Network (DQN)

### Model Architecture

The DQN model is a neural network with the following architecture:
- Input layer: Size equal to the state size (2)
- Hidden layers: Two fully connected layers with 24 neurons each
- Output layer: Size equal to the action size (3)

### Agent

The DQN agent interacts with the environment and learns the optimal policy using the following parameters:
- `gamma`: Discount factor (0.95)
- `epsilon`: Exploration rate (1.0, decaying to 0.01)
- `epsilon_decay`: Rate of decay for exploration (0.995)
- `learning_rate`: Learning rate for the optimizer (0.001)
- `memory`: Experience replay memory with a maximum length of 2000

## Training Loop

The training loop involves:
1. Resetting the environment to get the initial state.
2. For each time step:
   - The agent selects an action based on the current state using an epsilon-greedy policy.
   - The environment returns the next state, reward, and done flag based on the action.
   - The agent stores the experience in memory.
   - The agent samples a minibatch from memory and performs a training step using the sampled experiences.
3. Saving the model periodically (every 50 episodes).

## Usage

To run the training script, execute the following command in your terminal:

```bash
python acc_dqn.py
```

The script will train the agent for 1000 episodes and save the trained models periodically.

## Dependencies

- `numpy`
- `torch`
- `gym`
- `collections`
- `random`

Ensure you have all the required dependencies installed. You can install them using:

```bash
pip install numpy torch gym
```

## Code

```python
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gym
from gym import spaces
import numpy as np

class AdaptiveCruiseControlEnv(gym.Env):
    def __init__(self):
        super(AdaptiveCruiseControlEnv, self).__init__()
        self.max_speed = 30  # max speed of the vehicle
        self.min_distance = 5  # minimum safe distance
        self.max_distance = 100  # max distance to the car in front
        self.action_space = spaces.Discrete(3)  # actions: 0: decelerate, 1: maintain, 2: accelerate
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.max_speed, self.max_distance]), dtype=np.float32)

        self.reset()

    def reset(self):
        self.speed = np.random.uniform(0, self.max_speed)
        self.distance_to_lead_car = np.random.uniform(self.min_distance, self.max_distance)
        return np.array([self.speed, self.distance_to_lead_car], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.speed = max(0, self.speed - 1)
        elif action == 2:
            self.speed = min(self.max_speed, self.speed + 1)

        self.distance_to_lead_car = max(0, self.distance_to_lead_car - self.speed)
        reward = -1 if self.distance_to_lead_car < self.min_distance else 1

        done = self.distance_to_lead_car == 0
        state = np.array([self.speed, self.distance_to_lead_car], dtype=np.float32)
        return state, reward, done, {}

    def render(self, mode='human'):
        print(f'Speed: {self.speed}, Distance to lead car: {self.distance_to_lead_car}')

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
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
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

if __name__ == "__main__":
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
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % 50 == 0:
            agent.save(f"acc-dqn-{e}.pth")
```

This documentation provides a detailed explanation of the code and guides users on how to use and extend it.