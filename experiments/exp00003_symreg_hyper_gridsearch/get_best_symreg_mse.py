import os
import json
import sympy
import sys

exp_filepath = "/Users/garysaavedra/exp_output/symreg/exp00003_symreg_hyper_gridsearch"

true_equation = sympy.sympify("X0**4 + X0**3 + X0**2 + X0")
print("true equation: ", true_equation)

best_exps = []
best_mse = 100000000000000
best_mses = []
best_equations = []

get_best_for_nt = True
nt = 10

#loop through every subdir in exp_filepath
for subdir in os.listdir(exp_filepath):
    #open exp json file
    #put try except here
    try :
        with open(exp_filepath + "/" + subdir + "/output/results.json") as f:
            exp_data = json.load(f)

        with open(exp_filepath + "/" + subdir + "/output/args.json") as f:
            exp_args = json.load(f)
    except:
        print("Error with ", subdir)
        continue

    if get_best_for_nt and exp_args["num_train"] != nt:
        continue

    mse = exp_data["best_oob_fitness"][-1]
    if mse < best_mse:
        best_exps.append(subdir)
        best_mse = mse
        best_mses.append(mse)
        equation = sympy.simplify(sympy.sympify(exp_data["equation"]))
        best_equations.append(equation)

print("Best exps ", best_exps)
print("Best mses ", best_mses)
print("Best equations ", best_equations)
