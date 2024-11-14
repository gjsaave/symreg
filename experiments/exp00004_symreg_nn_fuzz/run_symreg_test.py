# import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# import copy

import torch
# import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import matplotlib.pyplot as plt
import argparse
import time
# import itertools
# import math

sys.path.append("../../")
from models import create_criterion
from datasets import ArrayData, generate_data
from logging_utils import save_dict, save_args
from train_test_methods import get_nn_output
from helper_methods import str2bool

from gplearn.genetic import SymbolicRegressor

import sympy

start = time.time()

# torch.set_printoptions(precision=4)
# torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='chaos code')
parser.add_argument('--epochs', type=int, default=1, metavar='N')
parser.add_argument('--datapath', type=str, default="/tmp", metavar='N',
                    help='where to find the numpy datasets')
parser.add_argument('--savepath', type=str, default="/tmp", metavar='N',
                    help='where to save off output')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default="rk1", metavar="N", help="which dataset to use. Note you may also need to set the datapath arg")
parser.add_argument('--loss_method', type=str, default="mse", metavar='N',
                    help='which criterion to use. choices are mse, nll, ce')
parser.add_argument('--debug', type=str2bool, default=False, help="for debugging. will run a small dataset")
parser.add_argument('--num_train', type=int, default=10, help='num train data points')
parser.add_argument('--num_val', type=int, default=10, help='num val data points')
parser.add_argument('--num_features', type=int, default=2, help="only used for tabular data")
parser.add_argument('--population', type=int, default=500, help='population size')
parser.add_argument('--generations', type=int, default=5, help='num generations')
parser.add_argument('--p_crossover', type=float, default=0.7, help='crossover probability')
parser.add_argument('--p_subtree_mutation', type=float, default=0.1, help='subtree mutation probability')
parser.add_argument('--p_point_mutation', type=float, default=0.1, help='subtree  point mutation probability')
parser.add_argument('--nngen', type=int, default=0, metavar="N", help="amount to generate data with NN. If 0 no data is generated")
parser.add_argument('--nngen_model_path', type=str, default="/tmp", help="path to model to load for nngen")


args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device ", device)

epochs = args.epochs
num_train = args.num_train
num_val = args.num_val

data_path = args.datapath
random_seed = args.seed
np.random.seed(random_seed)
dataset = args.dataset
num_features = args.num_features
loss_method = args.loss_method

population = args.population
p_crossover = args.p_crossover
p_subtree_mutation = args.p_subtree_mutation
p_point_mutation = args.p_point_mutation
generations = args.generations
nngen = args.nngen
nngen_model_path = args.nngen_model_path

savepath = args.savepath

exp_folder =  "sd" + str(random_seed) + str(dataset) + "_nt" + str(num_train) + "_nv" + str(num_val) + "_ep" + str(epochs) + "_pop" + str(population) + "_pc" + str(p_crossover) + "_psm" + str(p_subtree_mutation) + "_ppm" + str(p_point_mutation) + "_nf" + str(num_features) + "_gen" + str(generations) + "_ng" + str(nngen)

print(exp_folder)

output_savepath = os.path.join(savepath, exp_folder, "output")
if not os.path.exists(output_savepath):
    os.makedirs(output_savepath)

plots_savepath = os.path.join(savepath, exp_folder, "plots")
if not os.path.exists(plots_savepath):
    os.makedirs(plots_savepath)

#Datasets
data_train, data_train_labels, data_val, data_val_labels = generate_data(dataset, num_train, num_val, num_features, random_seed)

if nngen > 0:
    nngen_data_train, nngen_data_train_labels, _, _ = generate_data(dataset, nngen, 0, num_features, random_seed + 12362164) #Hack seed to generate unique features
    model = torch.load(os.path.join(nngen_model_path, "output/model.pt"), weights_only=False)

    batch_size = 1
    loss_method = "mse"
    ml_task = "reg"

    AllData = ArrayData(nngen_data_train, nngen_data_train_labels)
    all_loader = DataLoader(AllData, shuffle=False, batch_size=batch_size)

    criterion = create_criterion(loss_method, None, vx_term=False, dataset=dataset)
    _, _, nngen_outputs = get_nn_output(all_loader, model, criterion, batch_size, dataset, ml_task)

    data_train = np.concatenate((data_train, nngen_data_train), axis=0)
    data_train_labels = np.concatenate((data_train_labels, np.reshape(nngen_outputs, newshape=(-1, 1))), axis=0)

# print(data_train_labels.shape)
print("Num train points ", len(data_train))
print("Num val points ", len(data_val))

est_gp = SymbolicRegressor(population_size=population,
                           generations=generations,
                           p_crossover=p_crossover, p_subtree_mutation=p_subtree_mutation,
                           p_hoist_mutation=0.05, p_point_mutation=p_point_mutation, tournament_size=7,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0, metric=loss_method)
est_gp.fit(data_train, np.reshape(data_train_labels, (-1)))

# print(est_gp._program)
# print(est_gp.run_details_)

converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y
}

equation = sympy.sympify(str(est_gp._program), locals=converter)
print(str(equation))
# print(data_val.shape)

val_output = est_gp.predict(data_val)
#get mean squared error of val_output and data_val_labels
if loss_method == "mse":
    mse_val = np.mean((val_output - data_val_labels) ** 2)
else:
    raise ValueError("loss_method not supported")

#Add mse_val to dictionary
est_gp.run_details_["mse_val"] = mse_val
#Add program equation to dictionary
est_gp.run_details_["equation"] = str(equation)

save_dict(output_savepath, est_gp.run_details_)
save_args(output_savepath, args)

end = time.time()
print("Total run time: ", end - start)