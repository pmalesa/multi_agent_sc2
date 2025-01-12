import yaml

from envs.smacv2_env import make_smacv2_env
# from algorithms.dqn import DQNAgent
# from algorithms.vdn_qmix import VDNAgent, QMIXAgent

def evaluate(alg: str, config_path: str, checkpoint_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    env = make_smacv2_env(**config["env"])

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
    
    # agent.load(checkpoint_path)

    n_eval_episodes = config["evaluation"]["episodes"]
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # ...
            pass

        print(f"[EVAL] Episode {episode}: {episode_reward}")

    env.close()
    print(f"Evaluation complete.")
            
