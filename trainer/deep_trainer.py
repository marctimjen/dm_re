import time
import subprocess
import argparse

"""
This scipt is used to run the different model runs on either the cluster or a local machine. It is good if you have alot
of different models to test.
"""

parser = argparse.ArgumentParser(description='master process')
parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
parser.add_argument("-l", "--logs", required=True, help="Logs to load")  # which process log to load

args = parser.parse_args()

if args.user == "marc":
    py_path = "/home/hp/anaconda3/envs/lunar_lander/bin/python3"
    path = f"/home/hp/Documents/GitHub/dm_re/trainer/processes/process_{args.logs}.txt"
else:
    py_path = "/home/tyson/.conda/envs/lunar_lander/bin/python3"
    path = f"/home/tyson/dm_re/trainer/processes/process_{args.logs}.txt"


with open(path, "r") as f:
        file = []
        for line in f:
            file.append(line.strip("\n"))

commands = [[py_path] + i.split(" ") for i in file]

for command in commands:
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    time.sleep(1)
else:
    (output, err) = p.communicate()
    p_status = p.wait()
