import gymnasium as gym
from stable_baselines3 import DQN
import os
import time
import numpy as np

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

models_dir = "models/DQN"
logdir = "logs"

model_path = f"{models_dir}/18000000.zip"

model = DQN.load(model_path, env=env)

EPISODES = 1000

seeds_list = []

bad_eps = [6, 9, 12, 13, 25, 26, 27, 29, 35, 37, 44, 51, 52, 54, 66, 69, 75, 79, 85, 94, 107, 113, 116, 125, 126, 127, 129, 135, 136, 139, 140, 141, 146, 153, 156, 161, 164, 170, 174, 177, 178, 184, 195, 199, 200, 201, 204, 209, 224, 234, 237, 250, 252, 253, 266, 279, 294, 322, 337, 338, 339, 342, 348, 349, 352, 357, 360, 365, 367, 369, 377, 378, 388, 393, 398, 400, 401, 404, 409, 410, 411, 414, 425, 426, 430, 439, 445, 452, 453, 458, 460, 466, 468, 472, 478, 481, 484, 486, 498, 507, 514, 515, 522, 525, 526, 533, 534, 536, 537, 544, 548, 550, 555, 556, 559, 561, 562, 563, 568, 571, 582, 588, 597, 608, 614, 626, 627, 640, 642, 650, 661, 665, 672, 674, 677, 684, 686, 692, 702, 705, 717, 725, 735, 738, 740, 742, 755, 774, 775, 785, 790, 811, 814, 819, 821, 822, 828, 834, 840, 867, 868, 876, 886, 888, 889, 895, 897, 898, 900, 908, 914, 918, 922, 931, 940, 942, 944, 948, 949, 985, 994]


for ep in bad_eps: #range(EPISODES):
    print(ep)
    obs, _ = env.reset(seed=ep)
    save_seed = False
    done = False
    start_timer = time.time()

    while not done:
        env.render()

        t = time.time()

        if t - start_timer < 20:
            action, _ = model.predict(obs)
        else:
            action = np.array(0)
            print("Shutoff")
            save_seed = True

        obs, reward, done, _, _ = env.step(action)

    else:
        print(reward)
        if reward < 99.9 or save_seed:
            seeds_list.append(ep)

print(seeds_list)

env.close()
