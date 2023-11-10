import numpy as np
from fastapi import APIRouter
from models.dtos import LunarLanderPredictRequestDto, LunarLanderPredictResponseDto

import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN

env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode="human"
)

models_dir = "models/A2C"
model_path = f"{models_dir}/240000.zip"
model = A2C.load(model_path, env=env)

# models_dir = "models/PPO"
# model_path = f"{models_dir}/300000.zip"
# model = PPO.load(model_path, env=env)

# models_dir = "models/DQN"
# model_path = f"{models_dir}/2400000.zip"
# model = DQN.load(model_path, env=env)

router = APIRouter()


@router.post('/predict', response_model=LunarLanderPredictResponseDto)
def predict(request: LunarLanderPredictRequestDto):
    obs = request.observation

    if request.is_terminal:
        print("Current game is over, a new game will start with next request!")

    # Your moves go here!
    obs = np.array(obs)
    action, _ = model.predict(obs)

    return LunarLanderPredictResponseDto(
        action=action
    )
