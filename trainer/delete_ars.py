import neptune
import pandas
import numpy as np
import os

cv = int(167)

with open("NEPTUNE_API_TOKEN.txt", "r") as file:
    # Read the entire content of the file into a string
    token = file.read()

run = neptune.init_run(
    project="Kernel-bois/reinforcement-learning",
    api_token=token,
    with_id=f"REIN-{cv}",
)

m = run["reward/mean_reward"]
df = m.fetch_values()

delete_names = np.array(df.step[df.value < 289]).astype(int)

pa = "/home/tyson/dm_re/trainer/models"


for i in delete_names:
    path = pa + f"/ARS_REIN-{cv}_EP_" + str(i) + ".zip"
    os.remove(path)

