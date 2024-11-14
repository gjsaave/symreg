#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:06:02 2022

@author: gjsaave
"""

# import numpy as np
import sys
# from functools import partial
# from typing import Type, Any, Callable, Union, List, Optional
# import os
# import copy

import torch
import torch.nn as nn
# from torch import Tensor
# from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


# import matplotlib.pyplot as plt
# import math
from collections import OrderedDict


class NonLinearFC(nn.Module):
    def __init__(self, num_layers, num_features, num_classes, num_hidden_nodes, drop_rate, freeze=False):
        super().__init__()

        self.total_num_layers = num_layers
        self.drop_indices = []
        self.fc_indices = []
        self.conv_indices = []
        self.layers_to_coarsen = []
        self.layers_to_coarsen_conv = []
        
        m = OrderedDict()
        m["fc0"] = nn.Linear(num_features, num_hidden_nodes)
        m["drop0"] = DropoutCustom(p=drop_rate, freeze=freeze)
        m["relu0"] = nn.ReLU()
        self.drop_indices.append(0)
        self.fc_indices.append(0)
        self.layers_to_coarsen.append(0)
        
        for l in range(1,num_layers-1):
            m["fc" + str(l)] = nn.Linear(num_hidden_nodes, num_hidden_nodes)
            m["drop" + str(l)] = DropoutCustom(p=drop_rate, freeze=freeze)
            m["relu" + str(l)] = nn.ReLU()
            self.drop_indices.append(l)
            self.fc_indices.append(l)
            self.layers_to_coarsen.append(l)
            
        m["fc" + str(num_layers-1)] = nn.Linear(num_hidden_nodes, num_classes)
        m["drop" + str(num_layers-1)] = DropoutCustom(p=0, freeze=freeze)
        self.fc_indices.append(num_layers-1)
        self.drop_indices.append(num_layers-1)
        self.model = nn.ModuleDict(m)
        
    def forward(self, x):
        #This flattens the input if it is an image
        if len(x.shape) == 4:
            out = torch.flatten(x, start_dim=1)
        else:
            out = x
 
        for l in self.model.keys():
            out = self.model[l](out)
        
        return out

    
class DropoutCustom(torch.nn.Module):
    
    def __init__(self, p, freeze):
        """
        freeze: when set to True dropout will use the same self.drop_mat from previous forward call
        """
        super().__init__()
        self.p = p
        self.drop_mat = None
        self.freeze = freeze
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be a probability")
            
        self.scale_factor = (1 / (1 - self.p))
            
            
    def forward(self, x):
        if self.training and not self.freeze:
            self.drop_mat = torch.empty(x.size()[1]).uniform_(0, 1) >= self.p
            x = x.mul(self.drop_mat) * self.scale_factor

        elif self.training and self.freeze:
            x = x.mul(self.drop_mat) * self.scale_factor
            
        return x


class CustomMSE(nn.Module):
    def __init__(self, num_layers, vx_term):
        super(CustomMSE, self).__init__()
        #self.criterion = criterion
        self.vx_term = vx_term
        self.num_layers = num_layers
        self.mse_loss = torch.nn.MSELoss()

    def MSE(self, output, target):
        #loss = torch.mean((output - target)**2)
        loss = self.mse_loss(output, target)
        return loss

    def forward(self, output, target, x, wgrads, bgrads, level):
        #if we don't want an <v_bar, x_bar> term in the criterion
        if not self.vx_term:
            loss = self.MSE(output, target)

        #subtract <v, x> term from criterion
        else:
            #Note that v was compute on the previous level so we use level + 1
            v_trans_x = 0
            for l in range(self.num_layers):
                #Element-wise multiplication of v and x
                v_times_x_w = torch.mul(wgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1],  x.model["fc"+str(l)].weight)
                v_times_x_w_sum = torch.sum(v_times_x_w)
                v_times_x_b = torch.mul(bgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1], x.model["fc"+str(l)].bias)
                v_times_x_b_sum = torch.sum(v_times_x_b)
                v_trans_x = torch.add(v_trans_x, torch.add(v_times_x_w_sum, v_times_x_b_sum))

            #FIXME with more than 2 levels this may need to be a recursive call
            loss = self.MSE(output, target) - v_trans_x

        return loss


class CustomNLL(nn.Module):
    def __init__(self, num_layers, vx_term, dataset="peaks"):
        super(CustomNLL, self).__init__()
        #self.criterion = criterion
        self.vx_term = vx_term
        self.num_layers = num_layers
        self.dataset = dataset

    def nll(self, output, target):
        #Make target not one hot encoded
        if self.dataset == "peaks":
            _target = (target == 1).nonzero(as_tuple=False)[:,-1]
        else:
            _target = target

        loss = F.nll_loss(output, _target)
        return loss

    def forward(self, output, target, x, wgrads, bgrads, level):
        #if we don't want an <v_bar, x_bar> term in the criterion
        if not self.vx_term:
            loss = self.nll(output, target)

        #subtract <v, x> term from criterion
        else:
            #Note that v was compute on the previous level so we use level + 1
            v_trans_x = 0
            for l in range(self.num_layers):
                #Element-wise multiplication of v and x
                v_times_x_w = torch.mul(wgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1],  x.model["fc"+str(l)].weight)
                v_times_x_w_sum = torch.sum(v_times_x_w)
                v_times_x_b = torch.mul(bgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1], x.model["fc"+str(l)].bias)
                v_times_x_b_sum = torch.sum(v_times_x_b)
                v_trans_x = torch.add(v_trans_x, torch.add(v_times_x_w_sum, v_times_x_b_sum))


            #FIXME with more than 2 levels this may need to be a recursive call
            loss = self.nll(output, target) - v_trans_x

        return loss


class CustomCrossEntropy(nn.Module):
    def __init__(self, num_layers, vx_term, dataset="peaks"):
        super(CustomCrossEntropy, self).__init__()
        #self.criterion = criterion
        self.vx_term = vx_term
        self.num_layers = num_layers
        self.dataset = dataset
        self.ce_loss = nn.CrossEntropyLoss()

    def ce(self, output, target):
        #Make target not one hot encoded
        if self.dataset == "peaks":
            _target = (target == 1).nonzero(as_tuple=False)[:,-1]
        else:
            _target = target

        loss = self.ce_loss(output, _target)
        return loss

    def forward(self, output, target, x, wgrads, bgrads, level):
        #if we don't want an <v_bar, x_bar> term in the criterion
        if not self.vx_term:
            loss = self.ce(output, target)

        #subtract <v, x> term from criterion
        else:
            #Note that v was compute on the previous level so we use level + 1
            v_trans_x = 0
            for l in range(self.num_layers):
                #Element-wise multiplication of v and x
                v_times_x_w = torch.mul(wgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1],  x.model["fc"+str(l)].weight)
                v_times_x_w_sum = torch.sum(v_times_x_w)
                v_times_x_b = torch.mul(bgrads["level" + str(level+1)]["v_bar"]["fc"+str(l)][-1], x.model["fc"+str(l)].bias)
                v_times_x_b_sum = torch.sum(v_times_x_b)
                v_trans_x = torch.add(v_trans_x, torch.add(v_times_x_w_sum, v_times_x_b_sum))


            #FIXME with more than 2 levels this may need to be a recursive call
            loss = self.ce(output, target) - v_trans_x

        return loss


def create_optimizer(model, opt_method, base_lr, momentum, weight_decay, update_params):
    if opt_method == "sgd":
        optimizer = torch.optim.SGD(params=update_params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_method == "adagrad":
        optimizer = torch.optim.Adagrad(params=update_params, lr=base_lr, weight_decay=weight_decay)
    elif opt_method == "adam":
        optimizer = torch.optim.Adam(params=update_params, lr=base_lr, weight_decay=weight_decay)
    elif opt_method == "rmsprop":
        optimizer = torch.optim.RMSprop(params=update_params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer


def create_criterion(loss_method, total_num_layers, vx_term, dataset):
    if loss_method == "custommse":
        criterion = CustomMSE(num_layers=total_num_layers, vx_term=vx_term)
    elif loss_method == "customnll":
        criterion = CustomNLL(num_layers=total_num_layers, vx_term=vx_term, dataset=dataset)
    elif loss_method == "customce":
        criterion = CustomCrossEntropy(num_layers=total_num_layers, vx_term=vx_term, dataset=dataset)
    elif loss_method == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_method == "ce":
        criterion = torch.nn.CrossEntropyLoss()

    return criterion



class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)