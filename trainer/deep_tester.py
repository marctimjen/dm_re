import time

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from rl_pack.eval_policy import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import neptune
import numpy as np

n_tests = 2500

with open("NEPTUNE_API_TOKEN.txt", "r") as file:
    # Read the entire content of the file into a string
    token = file.read()

project = neptune.init_project(project="Kernel-bois/rl-models",
                               api_token=token)

models_dir = "models/"

while True:
    listed_models = os.listdir(models_dir)
    tested_models = project["names/"].fetch()
    tested_models = [i + ".zip" for i in tested_models.values()]

    for model_name in listed_models:
        if model_name in tested_models:
            continue
        else:
            model_name = model_name[:-4]
            model_path = models_dir + model_name

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

            if "PPO" in model_name:
                model = PPO.load(model_path, env=env)
            elif "A2C" in model_name:
                model = A2C.load(model_path, env=env)
            elif "DQN" in model_name:
                model = DQN.load(model_path, env=env)
            else:
                raise KeyError("Model name is not correct")

            run["model_name"] = model_name
            project[f"names/{model_name}"] = model_name
            project[f"models/{model_name}"].upload(model_path + ".zip")

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
            len_50 = np.quantile(episode_len, 0.5)
            len_mean = np.mean(episode_len)
            len_median = np.median(episode_len)

            run["reward/mean_reward"].log(m_rew)
            run["reward/std_reward"].log(std_rew)

            run["length/len_95"].log(len_95)
            run["length/len_mean"].log(len_mean)
            run["length/len_median"].log(len_median)

            env.close()
            run.stop()

    else:
        print("Close now")
        time.sleep(60)

project.stop()
