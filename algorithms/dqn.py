import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim = 128):
        super(QNetwork, self).__init__()
        self._layer_1 = nn.Linear(obs_dim, hidden_dim)
        self._layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self._layer_3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self._layer_1(x))
        x = F.relu(self._layer_2(x))
        q_values = self._layer_3(x)
        return q_values
    
class ReplayBuffer:
    def __init__(self, capacity):
        self._buffer = deque(maxlen = capacity)

    def add(self, obs, action, reward, next_obs, done):
        self._buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self._buffer, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return (
            torch.tensor(observations, dtype = torch.float32),
            torch.tensor(actions, dtype = torch.long),
            torch.tensor(rewards, dtype = torch.float32),
            torch.tensor(next_observations, dtype = torch.float32),
            torch.tensor(dones, dtype = torch.bool),
        )
    
    def __len__(self):
        return len(self._buffer)

class DQNAgent:
    def __init__(
            self,
            obs_dim = -1,
            state_dim = -1,
            n_actions = -1,
            n_agents = -1,
            learning_rate = 1e-3,
            gamma = 0.99,
            epsilon_start = 1.0,
            epsilon_end = 0.05,
            epsilon_decay = 10000,
            buffer_size = 50000,
            batch_size = 32,
            target_update_interval = 100,
            device = "cpu"
    ):
        
        if n_actions == -1:
            raise ValueError("Number of actions was not set correctly!")

        self._obs_dim = obs_dim
        self._state_dim = state_dim
        self._n_agents = n_agents
        self._n_actions = n_actions
        self._gamma = gamma
        self._batch_size = batch_size
        self._target_update_interval = target_update_interval
        self._device = device

        self._epsilon = epsilon_start
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._global_step = 0 # Value used to track epsilon decay process

        # Q-Network and Target Q-Network
        self._q_network = QNetwork(
            obs_dim = obs_dim * n_agents + state_dim,
            n_actions = n_actions * n_agents,
            hidden_dim = 800
        ).to(device)
        self._target_q_network = QNetwork(
            obs_dim = obs_dim * n_agents + state_dim,
            n_actions = n_actions * n_agents,
            hidden_dim = 800
        ).to(device)
        self._target_q_network.load_state_dict(self._q_network.state_dict())
        self._target_q_network.eval() # Set the module to evaluation mode (no training)

        self._optimizer = optim.Adam(self._q_network.parameters(), lr = learning_rate)

        self._replay_buffer = ReplayBuffer(buffer_size)

        self._loss_fn = nn.MSELoss()

    def select_actions(self, aggregated_obs: list, action_masks: list, exploit = True):
        # Decay epsilon
        self._global_step += 1
        if not exploit:
            if self._global_step < self._epsilon_decay:
                self._epsilon = self._epsilon_start - (self._global_step / self._epsilon_decay) * (self._epsilon_start - self._epsilon_end)
            else:
                self._epsilon =  self._epsilon_end
        else:
            self._epsilon = 0.0

        # Convert obs and action mask to pytorch tensors
        obs_tensor = torch.tensor(aggregated_obs, dtype = torch.float32).to(self._device)

        # Epsilon-greedy action choice
        actions = []
        if random.random() < self._epsilon and not exploit:
            # Choose random action
            for action_mask in action_masks:
                valid_actions = [i for i, valid in enumerate(action_mask) if valid]
                if not valid_actions:
                    actions.append(1)   # Default action if no valid action
                else:
                    actions.append(random.choice(valid_actions))
        else:
            # Choose action greedily
            with torch.no_grad():
                q_values = self._q_network(obs_tensor.unsqueeze(0)) # Shape: [1, n_actions]
                q_values = q_values.squeeze(0) # Shape: [n_actions]

                # Split Q-values for each agent
                # (total_actions = n_agents * n_actions)
                agent_q_values = q_values.view(self._n_agents, -1) # Shape: [n_agents, n_actions]
                for agent_id, action_mask in enumerate(action_masks):
                    mask = torch.tensor(action_mask, dtype = torch.bool).to(self._device)
                    agent_q_value = agent_q_values[agent_id]
                    agent_q_value[~mask] = -float('inf')
                    action = torch.argmax(agent_q_value).item()
                    actions.append(action)

        return actions
    
    def store_experience(self, aggregated_obs: torch.Tensor, actions: list, reward: float, next_aggregated_obs: torch.Tensor, done: bool):
        self._replay_buffer.add(aggregated_obs, actions, reward, next_aggregated_obs, done)

    def update(self):
        if len(self._replay_buffer) < self._batch_size:
            return 0.0
        
        # Sample experience from the replay buffer and perform a single gradient step
        aggregated_observations, actions, rewards, next_aggregated_observations, dones = self._replay_buffer.sample(self._batch_size)
        aggregated_observations = aggregated_observations.to(self._device)           # Shape: [batch_size, obs_dim]
        actions = actions.to(self._device)                                           # Shape: [batch_size, n_agents]
        rewards = rewards.to(self._device)                                           # Shape: [batch_size]
        next_aggregated_observations = next_aggregated_observations.to(self._device) # Shape: [batch_size, obs_dim]
        dones = dones.to(self._device)                                               # Shape: [batch_size]

        # Current Q-values
        q_values = self._q_network(aggregated_observations)                   # Shape: [batch_size, n_agents * n_actions]
        q_values = q_values.view(-1, self._n_agents, self._n_actions)         # Shape: [batch_size, n_agents, n_actions]

        # Gather Q-values for the taken actions
        actions = actions.long()                                              # Ensure actions are of type long
        actions = actions.unsqueeze(2)                                        # Shape: [batch_size, n_agents, 1]
        q_values = q_values.gather(2, actions).squeeze(2)                     # Shape: [batch_size, n_agents]

        # Target Q-values
        with torch.no_grad():
            next_q_values = self._target_q_network(next_aggregated_observations)                                # Shape: [batch_size, n_agents * n_actions]
            next_q_values = next_q_values.view(-1, self._n_agents, self._n_actions)                             # Shape: [batch_size, n_agents, n_actions]
            max_next_q_values, _ = next_q_values.max(dim = 2)                                                   # Shape: [batch_size, n_agents]
            target_q_values = rewards.unsqueeze(1) + self._gamma * max_next_q_values * (~dones.unsqueeze(1))    # Shape: [batch_size, n_agents]

        # Compute loss
        loss = self._loss_fn(q_values, target_q_values)

        # Optimize the Q-network
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Update target network
        if self._global_step % self._target_update_interval == 0:
            self._target_q_network.load_state_dict(self._q_network.state_dict())

        return loss.item()

    def save(self, path: str):
        # Save model parameters
        checkpoint = {
            "q_network": self._q_network.state_dict(),
            "target_q_network": self._target_q_network.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "global_step": self._global_step,
            "epsilon": self._epsilon
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        # Load model parameters
        checkpoint = torch.load(path, map_location = self._device)
        self._q_network.load_state_dict(checkpoint["q_network"])
        self._target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._global_step = checkpoint["global_step"]
        self._epsilon = checkpoint["epsilon"]
