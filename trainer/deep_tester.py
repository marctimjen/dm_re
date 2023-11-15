import time
import os
import neptune
import argparse
import subprocess

parser = argparse.ArgumentParser(description='master tester')
parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc', "hp"])  # this gives dir to data and save loc
parser.add_argument("-c", "--cuda", required=True, help="gpu number")

args = parser.parse_args()

cuda = int(args.cuda)

if cuda == 9:
    shifter = True
    cuda = -1
else:
    shifter = False

if args.user == "marc":
    py_path = "/home/marc/anaconda3/envs/lunar_lander/bin/python3"
    run_script = "/home/marc/Documents/GitHub/9semester/dm_re/trainer/run_test.py"
elif args.user == "hp":
    py_path = "/home/hp/anaconda3/envs/lunar_lander/bin/python3"
    run_script = "/home/hp/Documents/GitHub/dm_re/trainer/run_test.py"
else:
    py_path = "/home/tyson/.conda/envs/lunar_lander/bin/python3"
    run_script = "/home/tyson/dm_re/trainer/run_test.py"

n_tests = 1000

with open("NEPTUNE_API_TOKEN.txt", "r") as file:
    # Read the entire content of the file into a string
    token = file.read()

project = neptune.init_project(project="Kernel-bois/rl-models",
                               api_token=token)

models_dir = "models/"


while True:
    listed_models = os.listdir(models_dir)
    tested_models = project["names/"].fetch()
    tested_models = [i + ".zip" for i in tested_models.values()]

    not_tested_models = [mdl for mdl in listed_models if mdl not in tested_models]

    print(not_tested_models)

    j = 0
    for mdl in not_tested_models:
        if shifter:
            cuda += 1
            cuda %= 2

        command = [py_path, run_script, "-n", str(n_tests), "-m", str(mdl), "-c", str(cuda)]
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        j += 1
        time.sleep(1)

        if j % 1 == 0:
            (output, err) = p.communicate()
            p_status = p.wait()
            time.sleep(60)
    # for model_name in listed_models:
    # model_tester(mdl_name=model_name, models_dir=models_dir, token=token)

    print("Close now")
    time.sleep(10 * 60)
    print("too late")

project.stop()
