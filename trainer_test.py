import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import ARS, RecurrentPPO, TRPO
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

models_dir = "models/test"
logdir = "logs_test"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# model = PPO(
#     policy='MlpPolicy',
#     env=env,
#     n_steps=1024,
#     batch_size=64,
#     n_epochs=25,  # We're tuning this.
#     gamma=0.9908980966893566,  # We're tuning this.
#     gae_lambda=0.98,
#     ent_coef=0.01,
#     verbose=1,
#     tensorboard_log=logdir)

# model = ARS(policy='MlpPolicy', env=env)
# model = RecurrentPPO(policy='MlpLstmPolicy', env=env)
model = TRPO("MlpPolicy", env=env)


TIMESTPES = 500
EPISODES = 300
i = 1
for _ in range(2): #while True:
    model.learn(total_timesteps=TIMESTPES, reset_num_timesteps=False, tb_log_name="PPO_13")
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
