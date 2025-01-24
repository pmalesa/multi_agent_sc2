import yaml
import numpy as np
import torch

from envs.smacv2_env import make_smacv2_env
from algorithms.dqn import DQNAgent
from algorithms.qmix_vdn import QMIX_VDN
from tensordict import TensorDict
from torch.nn.utils import clip_grad_norm_

def evaluate(alg: str, config_path: str, checkpoint_path: str):
    # Initialization of the testing environment
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    env = make_smacv2_env(**config["env"])
    env_info = env.get_env_info()

    device = config["agent"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        config["agent"]["device"] = "cpu"
    print(f"Using device: {config['agent']['device']}")

    total_wins = 0
    rewards = []

    if alg == "dqn":
        n_agents = env_info["n_agents"]
        n_episodes = config["evaluation"]["episodes"]
        config["agent"]["state_dim"] = len(env.get_state())
        config["agent"]["obs_dim"] = env_info["obs_shape"] 
        config["agent"]["n_actions"] = env_info["n_actions"]
        config["agent"]["n_agents"] = n_agents

        dqn_agent = DQNAgent(**config["agent"])
        dqn_agent.load(checkpoint_path)

        for episode in range(1, n_episodes + 1):
            obs = env.reset()
            done = False
            episode_reward = 0

            while not done:
                obs = env.get_obs() # shape: [n_agents, observation_length]
                aggregated_obs = np.concatenate(obs, axis = 0)
                state = env.get_state()     
                aggregated_obs = np.concatenate([state, aggregated_obs], axis = 0)

                # Retrieve all action masks
                action_masks = [env.get_avail_agent_actions(agent_id) for agent_id in range(n_agents)]

                # Select actions for all agents
                actions = dqn_agent.select_actions(aggregated_obs, action_masks, exploit = False)
                
                # Perform actions in the environment
                reward, done, info = env.step(actions)
                episode_reward += reward
                rewards.append(episode_reward)

                if info["battle_won"] == True:
                    total_wins += 1

                # Aggregate next observations
                next_obs = env.get_obs()
                aggregated_next_obs = np.concatenate(next_obs, axis = 0)
                state = env.get_state()     
                aggregated_next_obs = np.concatenate([state, aggregated_next_obs], axis = 0)

            print(f"[EVALUATION] Episode {episode}: {episode_reward}")

    elif alg in ["vdn", "qmix"]: 
        env_info = env.get_env_info()
        n_agents = env_info["n_agents"]
        batch_size = config["agent"]["batch_size"]
        gamma = config["agent"]["gamma"]
        n_episodes = config["evaluation"]["episodes"]

        alg_settings = {"device" : device, "alg": alg, "minibatch": batch_size, "gamma": gamma, "tau": 0.005}
        qmix_vdn_agent = QMIX_VDN(env_info, alg_settings)
        qmix_vdn_agent.load(checkpoint_path)

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
                "agents": TensorDict({"observation": obs}, env_info["n_agents"]),
                "state": state,
                "next": TensorDict({
                    "agents": TensorDict({"observation": obs}),
                    "state": state,
                    "reward": 0 * torch.ones(1),
                    "done": (0) * torch.ones(1, dtype = torch.bool),
                    "terminated": (0) * torch.ones(1, dtype = torch.bool)
                })
            })

            while not terminated:
                obs = next_obs
                state = next_state
                avail_actions = env.get_avail_actions()
                td.set("mask", torch.BoolTensor(avail_actions).to(device))
                with torch.no_grad():
                    actions = qmix_vdn_agent.qnet(td)["agents"]["action"]
                reward, terminated, info = env.step(actions)

                rewards.append(episode_reward)
                if info["battle_won"] == True:
                    total_wins += 1

                next_obs = torch.tensor(np.array(env.get_obs())).to(device)
                next_state = torch.tensor(np.array(env.get_state())).to(device)
                td.set(("agents", "observation"), obs) 
                td.set("state", state)
                td.set(("next", "agents","observation"), next_obs)
                td.set(("next", "state"), next_state)
                td.set(("next", "reward"), reward * torch.ones(1))
                td.set(("next", "done"), (terminated) * torch.ones(1, dtype = torch.bool))
                td.set(("next", "terminated"), (terminated) * torch.ones(1, dtype = torch.bool))
                td.set(("next", "mask"), torch.BoolTensor(avail_actions).to(device))

                episode_reward += reward
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unknown algorithm: {alg}")
    
    avg_reward = np.mean(rewards) if rewards else 0.0
    win_rate = total_wins / n_episodes
    print(f"\n[{alg.upper()} RESULTS]\n    {total_wins}/{n_episodes} episodes won\n    {win_rate} win rate\n    Average episode reward: {avg_reward}\n")

    env.close()
    print(f"Evaluation complete.")
            
