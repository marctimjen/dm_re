import numpy as np
import time
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

# models_dir = "/home/paperspace/dm_re/models/A2C"
# model_path = f"{models_dir}/240000.zip"
# print(model_path)
# model = A2C.load(model_path, env=env)

models_dir = "/home/paperspace/dm_re/models/PPO"
model_path = f"{models_dir}/300000.zip"
model = PPO.load(model_path, env=env)

# models_dir = "/home/paperspace/dm_re/models/PPO"
# model_path = f"{models_dir}/1000000.zip"
# model = PPO.load(model_path, env=env)

# models_dir = "/home/paperspace/dm_re/models/DQN"
# model_path = f"{models_dir}/18000000.zip"
# model = DQN.load(model_path, env=env)

router = APIRouter()

start_time = time.time()
@router.post('/predict', response_model=LunarLanderPredictResponseDto)
def predict(request: LunarLanderPredictRequestDto):
    obs = request.observation

    if request.is_terminal:
        with open("validation_attempt.log", "a+") as f:
            f.write(f"{time.time()- start_time}: Ending game! \n")
        print("Current game is over, a new game will start with next request!")


    # Your moves go here!
    obs = np.array(obs)
    action, _ = model.predict(obs)

    return LunarLanderPredictResponseDto(
        action=action
    )
