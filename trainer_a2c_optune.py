import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import optuna
from optuna.samplers import TPESampler
# https://github.com/kingabzpro/deep-rl-class/blob/main/unit1/unit1_optuna_guide.ipynb

import neptune.integrations.optuna as optuna_utils
import neptune
import os

token = os.getenv("NEPTUNE_API_TOKEN")

run = neptune.init_run(
    project="Kernel-bois/reinforcement-learning",
    api_token=token,
)

run["param"] = {"re_algo": "a2c"}

neptune_callback = optuna_utils.NeptuneCallback(run)

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

n_envs = 16
# LunarLander environment with custom parameters
def make_env():
    custom_lunar_lander = gym.make('LunarLander-v2', continuous=False, gravity=-10.0, enable_wind=False,
                                   wind_power=15.0, turbulence_power=1.5)
    return custom_lunar_lander

# Create a vectorized environment with multiple instances
# vec_env = DummyVecEnv([lambda: custom_lunar_lander] * n_envs)
env = make_vec_env(make_env, n_envs=n_envs)


# env = gym.make(
#     "LunarLander-v2",
#     continuous=False,
#     gravity=-10.0,
#     enable_wind=False,
#     wind_power=15.0,
#     turbulence_power=1.5,
#
#     # render_mode="human"
# )

# env = make_vec_env(env_id=env, n_envs=16)

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
    model = A2C(
        policy='MlpPolicy',
        env=env,
        learning_rate=params["lr"],
        n_steps=params["n_steps"],
        gamma=params['gamma'],  # We're tuning this.
        gae_lambda=params['gae_lambda'],  # We're tuning this.
        ent_coef=0.01,
        verbose=verbose
    )
    model.learn(total_timesteps=params['total_timesteps'])  # We're tuning this.

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    score = mean_reward - std_reward

    if save_model:
        model.save(models_dir)

    return model, score, mean_reward, std_reward


def objective(trial):
    opt_id = trial.number
    params = {
        "n_steps": trial.suggest_int("n_steps", 10, 25),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3),
        "gamma": trial.suggest_float("gamma", 0.9900, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.95, 1.0),
        "total_timesteps": trial.suggest_int("total_timesteps", 1_000_000, 5_000_000)
    }
    model, score, mean_reward, std_reward = run_training(params)

    run[f"trials/trials/{opt_id}/reward/score"].log(score)
    run[f"trials/trials/{opt_id}/reward/mean_reward"].log(mean_reward)
    run[f"trials/trials/{opt_id}/reward/std_reward"].log(std_reward)
    return score


study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=500, callbacks=[neptune_callback])

print("Best trial score:", study.best_trial.values)
print("Best trial hyperparameters:", study.best_trial.params)


# from stable_baselines3.common.envs import LunarLander
#
# # LunarLander environment with custom parameters
# custom_lunar_lander = LunarLander(
#     continuous=False,
#     gravity=-10.0,
#     enable_wind=False,
#     wind_power=15.0,
#     turbulence_power=1.5,
# )