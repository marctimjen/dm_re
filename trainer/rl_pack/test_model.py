import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from .eval_policy import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import numpy as np
import neptune


def model_tester(mdl_name: str, models_dir: str, token: str, n_tests: int, project, cuda: int):

    cuda_device = f'cuda:{cuda}'

    m_name = mdl_name[:-4]
    model_path = models_dir + m_name
    run = neptune.init_run(
        project="Kernel-bois/rl-models",
        api_token=token,
    )
    env = Monitor(gym.make(
        "LunarLander-v2",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5
    ))
    if "PPO" in m_name:
        model = PPO.load(model_path, env=env, device=cuda_device)
    elif "A2C" in m_name:
        model = A2C.load(model_path, env=env, device=cuda_device)
    elif "DQN" in m_name:
        model = DQN.load(model_path, env=env, device=cuda_device)
    else:
        raise KeyError(f"Model name is not correct got {m_name}")
    run["model_name"] = m_name
    project[f"names/{m_name}"] = m_name
    project[f"models/{m_name}"].upload(model_path + ".zip")
    seeds = range(0, n_tests)
    episode_rewards = []
    episode_len = []
    for seed in seeds:
        env.reset(seed=seed)
        mean_reward, std_reward, episode_lengths = evaluate_policy(model, env, n_eval_episodes=1,
                                                                   deterministic=True)

        episode_rewards.append(mean_reward)
        episode_len.append(episode_lengths)

        assert len(episode_lengths) == 1, "More episode than 1 vas played"
        assert std_reward < 0.001, "std_reward is not 0"
    m_rew = np.mean(episode_rewards)
    std_rew = np.std(episode_rewards)
    episode_len = np.array(episode_len)
    len_95 = np.quantile(episode_len, 0.95)
    len_mean = np.mean(episode_len)
    len_median = np.median(episode_len)
    run["reward/mean_reward"].log(m_rew)
    run["reward/std_reward"].log(std_rew)
    run["length/len_95"].log(len_95)
    run["length/len_mean"].log(len_mean)
    run["length/len_median"].log(len_median)
    env.close()
    run.stop()
