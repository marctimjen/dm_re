from rl_pack.test_model import model_tester
import neptune
import argparse


parser = argparse.ArgumentParser(description='run tester')
parser.add_argument("-n", '--n_tests', required=True)
parser.add_argument("-m", '--mdl', required=True)
parser.add_argument("-c", "--cuda", required=True, help="gpu number")

args = parser.parse_args()

n_tests = int(args.n_tests)
mdl = str(args.mdl)
cuda = int(args.cuda)

models_dir = "models/"

with open("NEPTUNE_API_TOKEN.txt", "r") as file:
    # Read the entire content of the file into a string
    token = file.read()

project = neptune.init_project(project="Kernel-bois/rl-models",
                               api_token=token)

model_tester(mdl, models_dir, token, n_tests, project, cuda)
