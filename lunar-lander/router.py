import numpy as np
import time
import uuid
from fastapi import APIRouter
from models.dtos import LunarLanderPredictRequestDto, LunarLanderPredictResponseDto

import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import ARS, RecurrentPPO, TRPO

env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode="human"
)


model_path = f"/home/paperspace/dm_re/models/best_models/ARS_REIN-188_EP_2128.zip"
model = ARS.load(model_path, env=env)
router = APIRouter()

start_time = time.time()
@router.post('/predict', response_model=LunarLanderPredictResponseDto)
def predict(request: LunarLanderPredictRequestDto):
    obs = request.observation

    if request.is_terminal:
        print("Current game is over, a new game will start with next request!")

    # Your moves go here!
    obs = np.array(obs)
    action, _ = model.predict(obs, deterministic=True)

    return LunarLanderPredictResponseDto(
        action=action
    )
