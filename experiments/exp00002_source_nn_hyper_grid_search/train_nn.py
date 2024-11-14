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
from datasets import ArrayData, generate_data
from logging_utils import save_results, save_args
from train_test_methods import train, validation
from helper_methods import str2bool

start = time.time()

torch.set_printoptions(precision=4)
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='chaos code')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=1, metavar='N')
parser.add_argument('--datapath', type=str, default="/tmp", metavar='N',
                    help='where to find the numpy datasets')
parser.add_argument('--savepath', type=str, default="/tmp", metavar='N',
                    help='where to save off output')
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--num_hidden_nodes', type=int, default=8, help="num hidden")
parser.add_argument('--momentum', type=float, default=0, help="monentum")
parser.add_argument('--weight_decay', type=float, default=0, help="weight decay")
parser.add_argument('--num_layers', type=int, default=2, help="")
parser.add_argument('--drop_rate', type=float, default=0.5, help="drop rate for the fc layers. also controls the drop rate for the first coarse layer in mgdrop.")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_features', type=int, default=2, help="only used for tabular data")
parser.add_argument('--num_output_nodes', type=int, default=5)
parser.add_argument('--model_type', type=str, default="linear", metavar='N',
                    help='which model to use. Options are linear, nonlinear.')
parser.add_argument('--opt_method', type=str, default="sgd", metavar='N',
                    help='which optimizer to use. choices are sgd, adam, adagrad, rmsprop')
parser.add_argument('--dataset', type=str, default="rk1", metavar="N", help="which dataset to use. Note you may also need to set the datapath arg")
parser.add_argument('--loss_method', type=str, default="mse", metavar='N',
                    help='which criterion to use. choices are mse, nll, ce')
parser.add_argument('--debug', type=str2bool, default=False, help="for debugging. will run a small dataset")
parser.add_argument('--num_train', type=int, default=10, help='num train data points')
parser.add_argument('--num_val', type=int, default=10, help='num val data points')
parser.add_argument('--ml_task', type=str, default="reg", help="whether we are running classifcation or regression")
parser.add_argument('--update_str', type=str, default="all", help="which params to update")
parser.add_argument('--model_path', type=str, default="/tmp", help="path to model to load")
parser.add_argument('--tl_mode', type=str, default=None, help="what type of transfer learning to use")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device ", device)

epochs = args.epochs
num_features = args.num_features
num_output_nodes = args.num_output_nodes
num_hidden_nodes = args.num_hidden_nodes
momentum = args.momentum
batch_size = args.batch_size
weight_decay = args.weight_decay
base_lr = args.lr
num_layers=args.num_layers
opt_method=args.opt_method
loss_method = args.loss_method
drop_rate = args.drop_rate
num_train = args.num_train
num_val = args.num_val
ml_task = args.ml_task
update_str = args.update_str
model_path = args.model_path
tl_mode = args.tl_mode

data_path = args.datapath
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
model_type = args.model_type
dataset = args.dataset

savepath = args.savepath

exp_folder =  "sd" + str(random_seed) + "_bs" + str(batch_size) + "_mm" + str(momentum) + "_wd" + str(weight_decay) + "_nl" + str(num_layers) + "_lr" + str(base_lr) + "_dr" + str(drop_rate) + "_mt" + str(model_type) + "_nh" + str(num_hidden_nodes) + "_no" + str(num_output_nodes) + "_om" + str(opt_method) + "_ds" + str(dataset) + "_nc" + "_lm" + str(loss_method) + "_nt" + str(num_train) + "_nv" + str(num_val) + "_up" + str(update_str) + "_tl" + str(tl_mode) + "_ep" + str(epochs)

print(exp_folder)

output_savepath = os.path.join(savepath, exp_folder, "output")
if not os.path.exists(output_savepath):
    os.makedirs(output_savepath)

plots_savepath = os.path.join(savepath, exp_folder, "plots")
if not os.path.exists(plots_savepath):
    os.makedirs(plots_savepath)

#Datasets
data_train, data_train_labels, data_val, data_val_labels = generate_data(dataset, num_train, num_val, num_features, random_seed)

print("Num train points ", len(data_train))
print("Num val points ", len(data_val))
TrainData = ArrayData(data_train, data_train_labels)
ValData = ArrayData(data_val, data_val_labels)
train_loader = DataLoader(TrainData, shuffle=False, batch_size=batch_size)
val_loader = DataLoader(ValData, shuffle=False, batch_size=batch_size)

#Models
if model_type == "nonlinear":
    model = NonLinearFC(num_layers, num_features, num_output_nodes, num_hidden_nodes, drop_rate=drop_rate)
elif model_type == "load":
    model = torch.load(model_path, weights_only=False)

if tl_mode is None:
    pass
elif tl_mode == "final":
    layer_num_features = model.model["fc" + str(num_layers-1)].in_features
    model.model["fc" + str(num_layers-1)] = nn.Linear(layer_num_features, num_output_nodes)

model.to(device)

total_num_layers = model.total_num_layers
criterion = create_criterion(loss_method, total_num_layers, vx_term=False, dataset=dataset)

if update_str == "all":
    update_params = [param for param in model.parameters()]
elif update_str == "final":
    update_params = [model.model["fc" + str(num_layers-1)].weight, model.model["fc" + str(num_layers-1)].bias]

# for name, param in model.named_parameters():
#     print(name)
# print("update params ", update_params)
# print(list(model.parameters())[0].grad)
# sys.exit()

optimizer = create_optimizer(model, opt_method, base_lr, momentum, weight_decay, update_params)

#Train model
train_accs = []
val_accs = []
train_losses = []
val_losses = []
train_accs_all_iters = []
val_accs_all_iters = []
val_iters_list = []

val_acc, val_loss = validation(val_loader, model, criterion, batch_size, dataset, ml_task)
if ml_task == "class":
    print("initial val acc (untrained): ", val_acc)
elif ml_task == "reg":
    print("initial val los (untrained): ", val_loss.item())
# sys.exit()

full_idx = 0
for epoch in range(epochs):
    train_acc, train_loss, model = train(train_loader, model, criterion, optimizer, batch_size, dataset, ml_task)
    train_accs.append(train_acc)
    train_losses.append(train_loss.item())

    print("epoch: ", epoch)
    if ml_task == "class":
        print("train acc: ", train_acc)
    elif ml_task == "reg":
        print("train loss: ", train_loss.item())

    val_acc, val_loss = validation(val_loader, model, criterion, batch_size, dataset, ml_task)
    val_accs.append(val_acc)
    val_losses.append(val_loss.item())

    if ml_task == "class":
        print("val acc: ", val_acc)
    elif ml_task == "reg":
        print("val loss: ", val_loss.item())
    # train_acc_no_opt, _ = validation(train_loader, model, criterion, batch_size, dataset)


# val_acc, val_loss = validation(val_loader, model, criterion, batch_size, dataset)
# print("val acc: ", val_acc)

torch.save(model, output_savepath + "/model.pt")
save_results(output_savepath, epochs, train_accs, val_accs, train_losses, val_losses, train_accs_all_iters, val_accs_all_iters, val_iters_list)
save_args(output_savepath, args)

#print model params - for debugging
# for param in model.parameters():
#     print(param)
# for name, param in model.named_parameters():
#     print(name, param)
# print(list(model.parameters())[0].grad)

end = time.time()
print("Total run time: ", end - start)




