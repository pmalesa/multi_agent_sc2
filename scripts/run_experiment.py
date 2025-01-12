import yaml
import numpy as np
import time

from envs.smacv2_env import make_smacv2_env
# from algorithms.dqn import DQNAgent
# from algorithms.vdn_qmix import VDNAgent, QMIXAgent

def run_experiment(alg: str, config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    env = make_smacv2_env(**config["env"])
    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_episodes = config["training"]["episodes"]

    if alg == "dqn":
        # agent = DQNAgent(**config["agent"])
        pass
    elif alg == "vdn":
        # agent = VDNAgent(**config["agent"])
        pass
    elif alg == "qmix":
        # agent = QMIXAgent(**config["agent"])
        pass
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            obs = env.get_obs()
            state = env.get_state()
            # env.render()

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            
            reward, done, _ = env.step(actions)
            time.sleep(0.15)
            episode_reward += reward

        print(f"Episode {episode}: total_reward = {episode_reward}")

    env.close()
    print("Training complete.")
