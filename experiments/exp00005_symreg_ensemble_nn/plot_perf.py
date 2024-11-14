import json
import copy
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_results(seeds, nngens, param_name, params, pc, psm, ppm, generations, nt):
    curves = []
    for param in params:
        perf_per_train = []
        for nngen in nngens:
            perf_per_seed = []
            for seed in seeds:
                exp_filepath = "/Users/garysaavedra/exp_output/symreg/exp00005_symreg_ensemble_nn/sd" + str(seed) + "rk1_nt" + str(nt) + "_nv100_ep1_pop" + str(param) + "_pc" + str(pc) + "_psm" + str(psm) + "_ppm" + str(ppm) + "_nf1_gen" + str(generations) + "_ng" + str(nngen) + "_nmrandom" +  "/output"

                #open exp json file
                with open(exp_filepath + "/results.json") as f:
                    exp_data = json.load(f)

                #open exp args file
                with open(exp_filepath + "/args.json") as f:
                    exp_args = json.load(f)

                perf_per_seed.append(exp_data["best_oob_fitness"][-1])

            perf_per_train.append(copy.copy(perf_per_seed))

        curves.append(copy.copy(perf_per_train))

    return curves


def plot_results(curves):
    #for each curve in curves, plot the median and max and min spread
    for param_i in range(len(params)):
        perf_per_train = curves[param_i]
        perf_per_train = np.asarray(perf_per_train)
        median = np.median(perf_per_train, axis=1)
        max_val = np.max(perf_per_train, axis=1)
        min_val = np.min(perf_per_train, axis=1)
        plt.plot(nngen, median)
        plt.fill_between(nngen, min_val, max_val, alpha=0.1)

    # exp_filepath = "/Users/garysaavedra/exp_output/symreg/exp00003_symreg_hyper_gridsearch"

    plt.xlabel("Num NN generated points")
    plt.ylabel("Best oob fitness")
    plt.legend()
    plt.show()

seeds = [30, 31, 32, 33, 34]
nngen = [50, 100, 150, 200]
param_name = "pop"
params = [750]
pc = 0.5
psm = 0.1
ppm = 0.1
generations = 50
nt = 10
curves = get_results(seeds, nngen, param_name, params, pc, psm, ppm, generations, nt)
plot_results(curves)
