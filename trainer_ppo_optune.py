import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import optuna
from optuna.samplers import TPESampler
# https://github.com/kingabzpro/deep-rl-class/blob/main/unit1/unit1_optuna_guide.ipynb


env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    # render_mode="human"
)

# env = make_vec_env(gymenv, n_envs=16)

eval_env = Monitor(gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    # render_mode="human"
))

models_dir = "models/optuna/model"
logdir = "logs"

def run_training(params, verbose=0, save_model=False):
    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=params['n_epochs'],  # We're tuning this.
        gamma=params['gamma'],  # We're tuning this.
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=verbose
    )
    model.learn(total_timesteps=params['total_timesteps'])  # We're tuning this.

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    score = mean_reward - std_reward

    if save_model:
        model.save(models_dir)

    return model, score


def objective(trial):
  params = {
      "n_epochs": trial.suggest_int("n_epochs", 3, 5),
      "gamma": trial.suggest_uniform("gamma", 0.9900, 0.9999),
      "total_timesteps": trial.suggest_int("total_timesteps", 500_000, 2_000_000)
  }
  model, score = run_training(params)
  return score


study = optuna.create_study(sampler=TPESampler(), study_name="PPO-LunarLander-v2", direction="maximize")
study.optimize(objective, n_trials=10)

print("Best trial score:", study.best_trial.values)
print("Best trial hyperparameters:", study.best_trial.params)

# import os
#
# token = os.getenv("NEPTUNE_API_TOKEN")
#
# print(token)


