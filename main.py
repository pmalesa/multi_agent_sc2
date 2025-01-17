from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import replace

from smacv2.env import StarCraft2Env
import numpy as np
from absl import logging
import time
import torch
from tensordict import TensorDict
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from qmix_vdn_models import QMIX_VDN

logging.set_verbosity(logging.DEBUG)

def main():
    print("Running...")

    distribution_config = {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    }

    env = StarCraftCapabilityEnvWrapper(
        capability_config = distribution_config,
        map_name = "10gen_terran",
        debug = True,
        conic_fov = False,
        obs_own_pos = True,
        use_unit_ranges = True,
        min_attack_range = 2,
    )

    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_episodes = 200

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    alg_settings = {"device" : device, "alg": "qmix", "minibatch": 200, "gamma": 0.9, "tau": 0.005}
    alg = QMIX_VDN(env_info, alg_settings)
    lr = 5e-5
    optim = torch.optim.Adam(alg.loss_module.parameters(), lr)

    print("Training episodes")
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        reward = 0
        next_obs = torch.tensor(np.array(env.get_obs())).to(device)
        next_state = torch.tensor(np.array(env.get_state())).to(device)
        obs = next_obs
        state = next_state
        td = TensorDict({
        "agents": TensorDict({"observation": obs},env_info["n_agents"]),
        "state": state,
        "next": TensorDict({
            "agents": TensorDict({"observation": obs}),
            "state": state,
            "reward": 0*torch.ones(1),
            "done": (0)*torch.ones(1,dtype=torch.bool),
            "terminated": (0)*torch.ones(1,dtype=torch.bool)
        })
        })

        while not terminated:
            obs = next_obs
            state = next_state
            #env.render()
            avail_actions = env.get_avail_actions()
            td.set("mask", torch.BoolTensor(avail_actions).to(device)) 
            actions = alg.qnet_explore(td)["agents"]["action"]

            reward, terminated, a = env.step(actions)
            next_obs = torch.tensor(np.array(env.get_obs())).to(device)
            next_state = torch.tensor(np.array(env.get_state())).to(device)
            td.set(("agents","observation"), obs) 
            td.set("state", state)
            td.set(("next","agents","observation"), next_obs)
            td.set(("next","state"), next_state)
            td.set("reward", reward*torch.ones(1))
            td.set("done", (a["battle_won"])*torch.ones(1,dtype=torch.bool))
            td.set("terminated", (terminated and not a["battle_won"])*torch.ones(1,dtype=torch.bool))
            td.set(("next","mask"), torch.BoolTensor(avail_actions).to(device))

            alg.replay_buffer.extend(td.reshape(-1))
            subdata = alg.replay_buffer.sample()
            loss_vals = alg.loss_module(subdata)
            loss_value = loss_vals["loss"]
            loss_value.backward()
            optim.step()
            optim.zero_grad()
            alg.target_net_updater.step()
            time.sleep(0.15)
            episode_reward += reward
            


        print(f"Total reward in episode {e} = {episode_reward}")

    print("Finished.")

if __name__ == "__main__":
    main()
