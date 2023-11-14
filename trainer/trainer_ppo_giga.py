import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import neptune

with open("NEPTUNE_API_TOKEN.txt", "r") as file:
    # Read the entire content of the file into a string
    token = file.read()

run = neptune.init_run(
    project="Kernel-bois/reinforcement-learning",
    api_token=token,
)
run_id = run["sys/id"].fetch()

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
    n_steps=1024,
    batch_size=64,
    n_epochs=25,  # We're tuning this.
    gamma=0.9908980966893566,  # We're tuning this.
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=0
)

TIMESTPES = 10000
EPISODES = 3
i = 1
BEST_MEAN_REWARD = -1000


for i in range(EPISODES):  # True:
    model.learn(total_timesteps=TIMESTPES, reset_num_timesteps=False, tb_log_name="PPO_" + str(run_id))

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)

    run["reward/mean_reward"].log(mean_reward)
    run["reward/std_reward"].log(std_reward)

    if mean_reward > BEST_MEAN_REWARD:
        model.save(f"{models_dir}/PPO_{run_id}_EP_{i}")
        BEST_MEAN_REWARD = mean_reward
    else:
        BEST_MEAN_REWARD -= 1

    # i += 1


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
