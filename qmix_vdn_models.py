from __future__ import annotations

import time

import hydra
import torch

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import Categorical
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer, VDNMixer
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss


class QMIX_VDN:
    def __init__(self, env_settings, alg_settings):
        net = MultiAgentMLP(
                n_agent_inputs=env_settings["obs_shape"],
                n_agent_outputs=env_settings["n_actions"],
                n_agents=env_settings["n_agents"],
                centralised=False,
                share_params=True,
                device=alg_settings["device"],
                depth=2,
                num_cells=256,
                activation_class=nn.Tanh,
            )
        module = TensorDictModule(
            net, in_keys=[("agents", "observation")], out_keys=[("agents", "action_value")]
        )
        value_module = QValueModule(
            action_value_key=("agents", "action_value"),
            out_keys=[
                ("agents", "action"),
                ("agents", "action_value"),
                ("agents", "chosen_action_value"),
            ],
            action_space='categorical',
            spec=Categorical(env_settings["n_agents"])
        )
        self.qnet = SafeSequential(module, value_module)

        self.qnet_explore = TensorDictSequential(
            self.qnet,
            EGreedyModule(
                eps_init=0.3,
                eps_end=0,
                action_key=("agents", "action"),
                spec=Categorical(env_settings["n_agents"])
            ),
        )

        if alg_settings["alg"] == "qmix":
            self.mixer = TensorDictModule(
                module=QMixer(
                    state_shape=env_settings["state_shape"],
                    mixing_embed_dim=32,
                    n_agents=env_settings["n_agents"],
                    device=alg_settings["device"],
                ),
                in_keys=[("agents", "chosen_action_value"), ("agents", "observation")],
                out_keys=["chosen_action_value"],
            )
        elif alg_settings["alg"] == "vdn":
            self.mixer = TensorDictModule(
                module=VDNMixer(
                    n_agents=env_settings["n_agents"],
                    device=alg_settings["device"],
                ),
                in_keys=[("agents", "chosen_action_value")],
                out_keys=["chosen_action_value"],
            )
        else:
            raise ValueError("Mixer type not in the example")

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(4000, device=alg_settings["device"]),
            sampler=SamplerWithoutReplacement(),
            batch_size=alg_settings["minibatch"],
        )


        self.loss_module = QMixerLoss(self.qnet, self.mixer, delay_value=True)
        self.loss_module.set_keys(
            action_value=("agents", "action_value"),
            local_value=("agents", "chosen_action_value"),
            global_value="chosen_action_value",
            action=("agents", "action"),
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=alg_settings["gamma"])
        self.target_net_updater = SoftUpdate(self.loss_module, eps=1 - alg_settings["tau"])