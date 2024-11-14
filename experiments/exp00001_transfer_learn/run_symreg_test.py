import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import matplotlib.pyplot as plt
import argparse
import time
import itertools
import math

sys.path.append("../../")
from models import LinearFC, NonLinearFC, CustomMSE, NonLinearCNN, create_optimizer, create_criterion, FCActFunc, LogisticMap, init_weights_ones, LinearSharedWeight, CustomReLU, FCMultiActFunc, LogisticMapReLU
from datasets import ArrayData
# from logging_utils import save_results, save_args
from train_test_methods import train, validation, get_nn_output
from helper_methods import str2bool

from gplearn.genetic import SymbolicRegressor

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
parser.add_argument('--num_features', type=int, default=1)
parser.add_argument('--dataset', type=str, default="peaks", metavar="N", help="which dataset to use. Note you may also need to set the datapath arg")
parser.add_argument('--loss_method', type=str, default="mse", metavar='N',
                    help='which criterion to use. choices are mse, nll, ce')
parser.add_argument('--debug', type=str2bool, default=False, help="for debugging. will run a small dataset")
parser.add_argument('--train_amount', type=float, default=0.8, help='fraction of data to use for training')
parser.add_argument('--num_gen_data', type=int, default=100, help='for generated datasets, the number of totla data points generated')


args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device ", device)

epochs = args.epochs
num_features = args.num_features
train_amount = args.train_amount
num_gen_data = args.num_gen_data

data_path = args.datapath
random_seed = args.seed
np.random.seed(random_seed)
dataset = args.dataset

savepath = args.savepath

exp_folder =  "sd" + str(random_seed) + str(dataset) + "_ng" + str(num_gen_data)

output_savepath = os.path.join(savepath, exp_folder, "output")
if not os.path.exists(output_savepath):
    os.makedirs(output_savepath)

plots_savepath = os.path.join(savepath, exp_folder, "plots")
if not os.path.exists(plots_savepath):
    os.makedirs(plots_savepath)

#Datasets
if dataset == "add":
    data_all = np.random.uniform(low=0, high=1, size=(num_gen_data, num_features))
    data_all_labels = np.sum(data_all, axis=1)
    num_train_points = int(train_amount * num_gen_data)
    data_train = data_all[:num_train_points]
    data_train_labels = data_all_labels[:num_train_points]
    data_val = data_all[num_train_points:]
    data_val_labels = data_all_labels[num_train_points:]
elif dataset == "addwithnoise":
    data_all = np.random.uniform(low=0, high=1, size=(num_gen_data, num_features))
    data_all_labels = np.sum(data_all, axis=1) + np.random.normal(1, 0.1)
    num_train_points = int(train_amount * num_gen_data)
    data_train = data_all[:num_train_points]
    data_train_labels = data_all_labels[:num_train_points]
    data_val = data_all[num_train_points:]
    data_val_labels = data_all_labels[num_train_points:]
elif dataset == "modeladd":
    model_path = "/Users/garysaavedra/exp_output/symreg/exp00001_transfer_learn/sd30_bs8_mm0.0_wd0.0_nl2_lr0.1_dr0.0_mtnonlinear_nh8_no1_omadam_dsadd_nc_lmmse_ng1000/output/model.pt"
    model = torch.load(model_path, weights_only=False)
    model.eval()

    data_all = np.random.uniform(low=0, high=1, size=(num_gen_data, num_features))
    data_all_labels = np.sum(data_all, axis=1)
    AllData = ArrayData(data_all, data_all_labels)

    #TODO these are hardcoded right now
    batch_size = 1
    total_num_layers = 2
    loss_method = "mse"
    all_loader = DataLoader(AllData, shuffle=False, batch_size=batch_size)
    ml_task = "reg"

    criterion = create_criterion(loss_method, total_num_layers, vx_term=False, dataset=dataset)
    _, _, nn_outputs = get_nn_output(all_loader, model, criterion, batch_size, dataset, ml_task)

    data_train = data_all
    data_train_labels = nn_outputs

elif dataset == "modeladdmultiply":
    model_path = "/Users/garysaavedra/exp_output/symreg/exp00001_transfer_learn/sd30_bs8_mm0.0_wd0.0_nl2_lr0.1_dr0.0_mtload_nh8_no1_omadam_dsaddmultiply_nc_lmmse_ng10_upfinal_tlfinal/output/model.pt"
    model = torch.load(model_path, weights_only=False)
    model.eval()

    data_all = np.random.uniform(low=0, high=1, size=(num_gen_data, num_features))
    data_all_labels = data_all[:, 0] * 2 + data_all[:, 1]  # multiply first feature by a constant
    data_all_labels = np.reshape(data_all_labels, newshape=(-1, 1))
    AllData = ArrayData(data_all, data_all_labels)

    # TODO these are hardcoded right now
    batch_size = 1
    total_num_layers = 2
    loss_method = "mse"
    all_loader = DataLoader(AllData, shuffle=False, batch_size=batch_size)
    ml_task = "reg"

    criterion = create_criterion(loss_method, total_num_layers, vx_term=False, dataset=dataset)
    _, _, nn_outputs = get_nn_output(all_loader, model, criterion, batch_size, dataset, ml_task)

    data_train = data_all
    data_train_labels = nn_outputs

elif dataset == "rk1":
    data_all = np.random.uniform(low=0, high=1, size=(num_gen_data, num_features))
    # data_all[:, 0] = data_all[:, 0]*2 #multiply first feature by a constant
    data_all_labels = data_all[:, 0] + data_all[:, 1] ** 2 + data_all[:, 2] ** 3 + data_all[:, 3] ** 4
    data_all_labels = np.reshape(data_all_labels, newshape=(-1, 1))
    num_train_points = int(train_amount * num_gen_data)
    data_train = data_all[:num_train_points]
    data_train_labels = data_all_labels[:num_train_points]
    data_val = data_all[num_train_points:]
    data_val_labels = data_all_labels[num_train_points:]
elif dataset == "rk1t":
    data_all = np.random.uniform(low=0, high=1, size=(num_gen_data, num_features))
    # data_all[:, 0] = data_all[:, 0]*2 #multiply first feature by a constant
    data_all_labels = data_all[:, 0] + 2*data_all[:, 1] ** 2 + data_all[:, 2] ** 3 + data_all[:, 3] ** 4 + 2
    data_all_labels = np.reshape(data_all_labels, newshape=(-1, 1))
    # num_train_points = int(train_amount * num_gen_data)
    # data_train = data_all[:num_train_points]
    # data_train_labels = data_all_labels[:num_train_points]
    # data_val = data_all[num_train_points:]
    # data_val_labels = data_all_labels[num_train_points:]

    AllData = ArrayData(data_all, data_all_labels)

    model_path = "/Users/garysaavedra/exp_output/symreg/exp00001_transfer_learn/sd30_bs8_mm0.0_wd0.0_nl2_lr0.1_dr0.0_mtnonlinear_nh8_no1_omadam_dsrk1_nc_lmmse_ng63_upall_tlNone/output/model.pt"
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # TODO these are hardcoded right now
    batch_size = 1
    total_num_layers = 2
    loss_method = "mse"
    all_loader = DataLoader(AllData, shuffle=False, batch_size=batch_size)
    ml_task = "reg"

    criterion = create_criterion(loss_method, total_num_layers, vx_term=False, dataset=dataset)
    _, _, nn_outputs = get_nn_output(all_loader, model, criterion, batch_size, dataset, ml_task)

    data_train = data_all
    data_train_labels = nn_outputs



print("Num train points ", len(data_train))
# print("Num val points ", len(data_val))

est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(data_train, data_train_labels)

print(est_gp._program)

#Train model
# train_accs = []
# val_accs = []
# train_losses = []
# val_losses = []
# train_accs_all_iters = []
# val_accs_all_iters = []
# val_iters_list = []

# save_results(output_savepath, epochs, train_accs, val_accs, train_losses, val_losses, train_accs_all_iters, val_accs_all_iters, val_iters_list)
# save_args(output_savepath, args)

end = time.time()
print("Total run time: ", end - start)