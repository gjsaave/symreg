import os
import json
import sympy
import sys

exp_filepath = "/Users/garysaavedra/exp_output/symreg/exp00002_source_nn_hyper_grid_search"

best_exp = None
best_mse = 100000000000000
best_equations = []
num_train = 10
#loop through every subdir in exp_filepath
for subdir in os.listdir(exp_filepath):
    #open exp json file
    try :
        with open(exp_filepath + "/" + subdir + "/output/results.json") as f:
            exp_data = json.load(f)

        with open(exp_filepath + "/" + subdir + "/output/args.json") as f:
            exp_args = json.load(f)
    except:
        print("Error with ", subdir)
        continue

    if exp_args["num_train"] != num_train:
        continue

    mse = exp_data["val_losses"][-1]
    if mse < best_mse:
        best_exp = subdir
        best_mse = mse

print("Best exp ", best_exp)
print("Best mse ", best_mse)

