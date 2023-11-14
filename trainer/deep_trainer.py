import time
import subprocess
import argparse

"""
This scipt is used to run the different model runs on either the cluster or a local machine. It is good if you have alot
of different models to test.
"""

parser = argparse.ArgumentParser(description='master process')
parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
parser.add_argument("-g", '--gange', help="Number of processes to run", default=1, type=int)
args = parser.parse_args()

if args.user == "marc":
    py_path = "/home/marc/anaconda3/envs/lunar_lander/bin/python3"
    commands = ["/home/marc/Documents/GitHub/9semester/dm_re/trainer/trainer_ppo_giga.py"]
else:
    py_path = "/home/cv05f23/.conda/envs/weak/bin/python3"
    commands = ["/home/marc/Documents/GitHub/9semester/dm_re/trainer/trainer_ppo_giga.py"]

commands *= args.gange

for com in commands:
    command = [py_path, com]

    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    time.sleep(1)
else:
    (output, err) = p.communicate()
    p_status = p.wait()
