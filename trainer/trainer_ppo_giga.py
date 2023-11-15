import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import neptune
import copy
import argparse
import parameter_settings

parser = argparse.ArgumentParser(description='PPO_giga_trainer')
parser.add_argument("-c", "--cuda", required=True, help="gpu number")
parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load

args = parser.parse_args()
cuda_device = f'cuda:{args.cuda}'
params = parameter_settings.HYPERPARAMS[args.params]


with open("NEPTUNE_API_TOKEN.txt", "r") as file:
    # Read the entire content of the file into a string
    token = file.read()

run = neptune.init_run(
    project="Kernel-bois/reinforcement-learning",
    api_token=token,
)
run_id = run["sys/id"].fetch()

run["params"] = params | {"param_name": str(args.params)}

env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

eval_env = Monitor(gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5
))
eval_env.reset()
env.reset()

models_dir = "models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model = PPO(
    "MlpPolicy",
    env=env,
    device=cuda_device,
    n_steps=params["n_steps"],
    batch_size=params["batch_size"],
    n_epochs=params["n_epochs"],
    gamma=params["gamma"],
    gae_lambda=params["gae_lambda"],
    ent_coef=params["ent_coef"],
    verbose=0
)

TIMESTPES = 20000
BEST_MEAN_REWARD = 280
i = 0
while True:
    model.learn(total_timesteps=TIMESTPES, reset_num_timesteps=False)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)

    run["reward/mean_reward"].log(mean_reward)
    run["reward/std_reward"].log(std_reward)

    if mean_reward > BEST_MEAN_REWARD:
        model.save(f"{models_dir}/PPO_{run_id}_EP_{i}")

    i += 1


# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#
#     while not done:
#         env.render()
#         obs, reward, done, info = env.step(env.action_space.sample())


env.close()
eval_env.close()
run.stop()
