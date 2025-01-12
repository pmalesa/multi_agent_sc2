import torch
import torch.nn
import torch.optim as optim

from torchrl.modulse import MLP
from torchrl.objectives import DQNLoss
from torchrl.data import TensorDictReplayBuffer, TensorDict, CompositeSpec

class DQNAgent:
    def __init__(
            self,
            obs_dim = 64,
            n_actions = 10,
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
        
        self._obs_dim = obs_dim
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

        self._q_network = MLP(
            in_features = obs_dim,
            out_features = n_actions,
            depth = 2,
            num_cells = 128
        ).to(device)

        self._target_q_network = MLP(
            in_features = obs_dim,
            out_features = n_actions,
            depth = 2,
            num_cells = 128
        ).to(device)

        # Initialize target network's weiths
        self._target_q_network.load_state_dict(self._q_network.state_dict())

        self._optimizer = optim.Adam(self._q_network.parameters(), lr = learning_rate)

        # Replay buffer (and its specification) initialization
        self._buffer_spec = CompositeSpec(
            obs = torch.Size([obs_dim]),
            action = torch.Size([]),
            reward = torch.Size([]),
            next_obs = torch.Size([obs_dim]),
            done = torch.Size([])
        )
        self._replay_buffer = TensorDictReplayBuffer(
            capacity = buffer_size,
            storage = self._buffer_spec
        )

# TODO ...