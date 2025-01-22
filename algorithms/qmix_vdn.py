from __future__ import annotations

from tensordict.nn import TensorDictModule, TensorDictSequential
import torch
from torch import nn
from torchrl.data import Categorical
from torchrl._utils import logger as torchrl_logger
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
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
            action_mask_key="mask",
            action_space='categorical',
            # spec=Categorical(env_settings["n_agents"])
        )
        self.qnet = SafeSequential(module, value_module)

        self.qnet_explore = TensorDictSequential(
            self.qnet,
            EGreedyModule(
                eps_init=1.0,
                eps_end=0.05,
                action_key=("agents", "action"),
                spec=Categorical(env_settings["n_actions"]),
                annealing_num_steps=3000/2*50,
                action_mask_key="mask",
            ),
        )

        if alg_settings["alg"] == "qmix":
            self.mixer = TensorDictModule(
                module=QMixer(
                    state_shape=(env_settings["state_shape"],),
                    mixing_embed_dim=32,
                    n_agents=env_settings["n_agents"],
                    device=alg_settings["device"],
                ),
                in_keys=[("agents", "chosen_action_value"), "state"],
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
            storage=LazyTensorStorage(50000, device=alg_settings["device"]),
            sampler=SamplerWithoutReplacement(),
            batch_size=alg_settings["minibatch"],
        )


        self.loss_module = QMixerLoss(self.qnet, self.mixer, delay_value=True, action_space='categorical')
        self.loss_module.set_keys(
            action_value=("agents", "action_value"),
            local_value=("agents", "chosen_action_value"),
            global_value="chosen_action_value",
            action=("agents", "action"),
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=alg_settings["gamma"])
        self.target_net_updater = SoftUpdate(self.loss_module, eps=1 - alg_settings["tau"])

    def save(self, path: str):
        # Save model parameters
        checkpoint = {
            'qnet_state_dict': self.qnet.state_dict(),
            'mixer_state_dict': self.mixer.state_dict(),
            'qnet_explore_state_dict': self.qnet_explore.state_dict(),
            'loss_module_state_dict': self.loss_module.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        # Load model parameters
        state = torch.load(path, map_location = self.qnet.device)
        self.qnet.load_state_dict(state['qnet_state_dict'])
        self.mixer.load_state_dict(state['mixer_state_dict'])
        self.qnet_explore.load_state_dict(state['qnet_explore_state_dict'])
        self.loss_module.load_state_dict(state['loss_module_state_dict'])
