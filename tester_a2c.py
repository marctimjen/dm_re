import gymnasium as gym
from stable_baselines3 import A2C
import os

env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode="human"
)

env.reset()

models_dir = "models/A2C"
logdir = "logs"

model_path = f"{models_dir}/240000.zip"

model = A2C.load(model_path, env=env)

EPISODES = 10

for er in range(EPISODES):
    obs = env.reset()
    done = False
    first = True
    while not done:
        env.render()
        if first:
            action, _ = model.predict(obs[0])
            first = False
        else:
            action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

env.close()
