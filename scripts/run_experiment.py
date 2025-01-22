import yaml
import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt

from envs.smacv2_env import make_smacv2_env
from algorithms.dqn import DQNAgent
# from algorithms.vdn_qmix import VDNAgent, QMIXAgent

def run_experiment(alg: str, config_path: str):
    # Initialization of the training environment
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    rewards_file = f"{alg}_rewards.txt"
    loss_file = f"{alg}_loss.txt"
    with open(rewards_file, 'w') as file:
        pass
    with open(loss_file, 'w') as file:
        pass

    env = make_smacv2_env(**config["env"])
    env_info = env.get_env_info()

    n_agents = env_info["n_agents"]
    n_episodes = config["training"]["episodes"]
    config["agent"]["state_dim"] = len(env.get_state())
    config["agent"]["obs_dim"] = env_info["obs_shape"] 
    config["agent"]["n_actions"] = env_info["n_actions"]
    config["agent"]["n_agents"] = n_agents

    device = config["agent"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        config["agent"]["device"] = "cpu"
    print(f"Using device: {config['agent']['device']}")

    # Training process
    if alg == "dqn":
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
        model_save_path = os.path.join(checkpoint_path, f"{alg}_checkpoint.pt")
        dqn_agent.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    elif alg == "vdn":
        # agent = VDNAgent(**config["agent"])
        pass
    elif alg == "qmix":
        # agent = QMIXAgent(**config["agent"])
        pass
    else:
        raise ValueError(f"Unknown algorithm: {alg}")
    
    losses = read_results(f"{alg}_loss.txt")
    rewards = read_results(f"{alg}_rewards.txt")
    plot_results(losses, rewards, alg)

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

def plot_results(loss_values, reward_values, alg):
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
        plt.savefig(f"{alg}_loss.png")
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
        plt.savefig(f"{alg}_reward.png")
        plt.close()
