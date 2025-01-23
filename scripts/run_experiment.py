import yaml
import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt

from envs.smacv2_env import make_smacv2_env
from algorithms.dqn import DQNAgent
from algorithms.qmix_vdn import QMIX_VDN
from absl import logging
from tensordict import TensorDict
from torch.nn.utils import clip_grad_norm_

def run_experiment(alg: str, config_path: str):
    # Initialization of the training environment
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    os.makedirs("results", exist_ok = True)
    map_name = config["env"]["map_name"]
    rewards_file = f"results/{alg}_rewards{map_name}.txt"
    loss_file = f"results/{alg}_loss{map_name}.txt"
    with open(rewards_file, 'w') as file:
        pass
    with open(loss_file, 'w') as file:
        pass

    env = make_smacv2_env(**config["env"])
    env_info = env.get_env_info()

    device = config["agent"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        config["agent"]["device"] = "cpu"
    print(f"Using device: {config['agent']['device']}")

    # Training process
    if alg == "dqn":
        n_agents = env_info["n_agents"]
        n_episodes = config["training"]["episodes"]
        config["agent"]["state_dim"] = len(env.get_state())
        config["agent"]["obs_dim"] = env_info["obs_shape"] 
        config["agent"]["n_actions"] = env_info["n_actions"]
        config["agent"]["n_agents"] = n_agents

        dqn_agent = DQNAgent(**config["agent"])

        for episode in range(1, n_episodes + 1):
            obs = env.reset()
            done = False
            episode_reward = 0
            losses = []

            while not done:
                obs = env.get_obs()         # shape: [n_agents, observation_length]
                aggregated_obs = np.concatenate(obs, axis = 0)
                state = env.get_state()     
                aggregated_obs = np.concatenate([state, aggregated_obs], axis = 0)
                loss = 0.0

                # Retrieve all action masks
                action_masks = [env.get_avail_agent_actions(agent_id) for agent_id in range(n_agents)]

                # Select actions for all agents
                actions = dqn_agent.select_actions(aggregated_obs, action_masks, exploit = False)
                
                # Perform actions in the environment
                reward, done, _ = env.step(actions)
                episode_reward += reward

                # Aggregate next observations
                next_obs = env.get_obs()
                aggregated_next_obs = np.concatenate(next_obs, axis = 0)
                state = env.get_state()     
                aggregated_next_obs = np.concatenate([state, aggregated_next_obs], axis = 0)
                
                # Store experience
                dqn_agent.store_experience(aggregated_obs, actions, reward, aggregated_next_obs, done)

                # Update the agent after each episode
                if episode > 50:
                    loss = dqn_agent.update()
                    if loss is not None:
                        losses.append(loss)

            avg_loss = np.mean(losses) if losses else 0.0

            # ---------- SAVE REWARDS ----------
            print(f"Episode {episode}: total_reward = {episode_reward}, avg_loss = {avg_loss}")
            with open(rewards_file, 'a') as file:
                file.write(f"{episode_reward}\n")
            # ---------- SAVE LOSS ----------
            with open(loss_file, 'a') as file:
                file.write(f"{avg_loss}\n")
            # -------------------------------
        
        # Save model
        checkpoint_path = config["training"].get("checkpoint_path", "checkpoints/")
        os.makedirs(checkpoint_path, exist_ok = True)
        model_save_path = os.path.join(checkpoint_path, f"{alg}_checkpoint{map_name}.pt")
        dqn_agent.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    elif alg in ["vdn", "qmix"]: 
        env_info = env.get_env_info()
        n_actions = env_info["n_actions"]
        n_agents = env_info["n_agents"]
        lr = config["agent"]["learning_rate"]
        batch_size = config["agent"]["batch_size"]
        gamma = config["agent"]["gamma"]
        target_update_interval = config["agent"]["target_update_interval"]
        n_episodes = config["training"]["episodes"]

        alg_settings = {"device" : device, "alg": alg, "minibatch": batch_size, "gamma": gamma, "tau": 0.005}
        qmix_vdn_agent = QMIX_VDN(env_info, alg_settings)

        optim = torch.optim.Adam(qmix_vdn_agent.loss_module.parameters(), lr)

        i = 0
        for e in range(n_episodes):
            env.reset()
            terminated = False
            episode_reward = 0
            reward = 0
            next_obs = torch.tensor(np.array(env.get_obs())).to(device)
            next_state = torch.tensor(np.array(env.get_state())).to(device)
            obs = next_obs
            state = next_state
            losses = []
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
                # env.render(None)
                avail_actions = env.get_avail_actions()
                td.set("mask", torch.BoolTensor(avail_actions).to(device))
                with torch.no_grad():
                    actions = qmix_vdn_agent.qnet_explore(td)["agents"]["action"]
                reward, terminated, a = env.step(actions)
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
                
                qmix_vdn_agent.replay_buffer.extend(td.reshape(-1))
                i += 1
                loss = 0.0
                if e >= 50:
                    subdata = qmix_vdn_agent.replay_buffer.sample()
                    loss_vals = qmix_vdn_agent.loss_module(subdata)
                    loss_value = loss_vals["loss"]
                    loss += loss_value.item()
                    loss_value.backward()
                    clip_grad_norm_(qmix_vdn_agent.loss_module.parameters(), 1)
                    optim.step()
                    optim.zero_grad()

                episode_reward += reward
                if loss is not None:
                    losses.append(loss)

                if (i) % target_update_interval == 0:
                    qmix_vdn_agent.target_net_updater.step()
                torch.cuda.empty_cache()
                    
            avg_loss = np.mean(losses) if losses else 0.0

            # ---------- SAVE REWARDS ----------
            print(f"Episode {e}: total_reward = {episode_reward}, avg_loss = {avg_loss}")
            with open(rewards_file, 'a') as file:
                file.write(f"{episode_reward}\n")
            # ---------- SAVE LOSS ----------
            with open(loss_file, 'a') as file:
                file.write(f"{avg_loss}\n")
            # -------------------------------

        # Save model
        checkpoint_path = config["training"].get("checkpoint_path", "checkpoints/")
        os.makedirs(checkpoint_path, exist_ok = True)
        model_save_path = os.path.join(checkpoint_path, f"{alg}_checkpoint{map_name}.pt")
        qmix_vdn_agent.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    else:
        raise ValueError(f"Unknown algorithm: {alg}")
    
    losses = read_results(f"results/{alg}_loss{map_name}.txt")
    rewards = read_results(f"results/{alg}_rewards{map_name}.txt")
    plot_results(losses, rewards, alg, map_name)

    env.close()
    print("Training complete.")

def read_results(file_path: str):
    values = []
    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start = 1):
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        value = float(stripped_line)
                        values.append(value)
                    except ValueError:
                        print(f"Found invalid number in line {line_number} in file '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except IOError as e:
        print(f"An I/O error occurred while reading '{file_path}': {e}")
        return None

    if not values:
        print(f"No valid float values found in '{file_path}'.")
        return None

    return values

def plot_results(loss_values, reward_values, alg, map_name):
    if loss_values:
        plt.figure(figsize = (12, 6))
        x_loss = np.arange(1, len(loss_values) + 1)
        plt.plot(x_loss, loss_values, linestyle = '-', color = 'r', label = 'Loss')
        plt.title(f"Loss Over Episodes for {alg.upper()}")
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{alg}_loss{map_name}.png")
        plt.close()

    if reward_values:
        plt.figure(figsize = (12, 6))
        x_reward = np.arange(1, len(reward_values) + 1)
        plt.plot(x_reward, reward_values, linestyle = '-', color=  'g', label = 'Reward')
        plt.title(f"Reward Over Episodes for {alg.upper()}")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{alg}_reward{map_name}.png")
        plt.close()
