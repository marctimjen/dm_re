import gymnasium as gym
from stable_baselines3 import DQN
import os

env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

env.reset()

models_dir = "modelsnew/DQN"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTPES = 10000
EPISODES = 300
i = 1
while True:
    model.learn(total_timesteps=TIMESTPES, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTPES * i}")
    i += 1

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#
#     while not done:
#         env.render()
#         obs, reward, done, info = env.step(env.action_space.sample())


env.close()
