import json
import copy
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_results(seeds, num_trains, param_name, params, pc, psm, ppm, generations):
    curves = []
    for param in params:
        perf_per_train = []
        for nt in num_trains:
            perf_per_seed = []
            for seed in seeds:
                exp_filepath = "/Users/garysaavedra/exp_output/symreg/exp00003_symreg_hyper_gridsearch_bak/sd" + str(seed) + "rk1_nt" + str(nt) + "_nv100_ep1_pop" + str(param) + "_pc" + str(pc) + "_psm" + str(psm) + "_ppm" + str(ppm) + "_nf1_gen" + str(generations) + "/output"

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
        plt.plot(num_trains, median, label=param_name + "=" + str(params[param_i]))
        plt.fill_between(num_trains, min_val, max_val, alpha=0.1)

        print(median)

    plt.xlabel("Num train points")
    plt.ylabel("Best oob fitness")
    plt.legend()
    plt.show()

# seeds = [30, 31, 32, 33, 34]
# num_trains = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# param_name = "pop"
# params = [250, 500, 750]
# pc = 0.5
# psm = 0.05
# ppm = 0.1
# generations = 10
# curves = get_results(seeds, num_trains, param_name, params, pc, psm, ppm, generations)
# plot_results(curves)
#
# seeds = [30, 31, 32, 33, 34]
# num_trains = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# param_name = "pop"
# params = [250, 500, 750]
# pc = 0.5
# psm = 0.15
# ppm = 0.1
# generations = 50
# curves = get_results(seeds, num_trains, param_name, params, pc, psm, ppm, generations)
# plot_results(curves)

seeds = [30, 31, 32, 33, 34]
num_trains = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param_name = "pop"
params = [750]
pc = 0.7
psm = 0.05
ppm = 0.1
generations = 10
curves = get_results(seeds, num_trains, param_name, params, pc, psm, ppm, generations)
plot_results(curves)