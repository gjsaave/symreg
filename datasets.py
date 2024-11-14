#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:05:22 2022

@author: gjsaave
"""

import numpy as np
import sys
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
import os
import copy

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import math


def generate_data(dataset, num_train, num_val, num_features, random_seed):
    if dataset == "rk1":
        num_gen_data = num_train + num_val
        data_all = np.random.uniform(low=-1, high=1, size=(num_gen_data, num_features))
        # data_all[:, 0] = data_all[:, 0]*2 #multiply first feature by a constant
        data_all_labels = data_all[:, 0] + data_all[:, 0] ** 2 + data_all[:, 0] ** 3 + data_all[:, 0] ** 4
        data_all_labels = np.reshape(data_all_labels, newshape=(-1, 1))
        data_train = data_all[:num_train]
        data_train_labels = data_all_labels[:num_train]
        data_val = data_all[num_train:]
        data_val_labels = data_all_labels[num_train:]
    elif dataset == "rk1t":
        np.random.seed(
            random_seed + 1231738)  # Hack. The ensures that transfer learn input features are unique from source feature values
        num_gen_data = num_train + num_val
        data_all = np.random.uniform(low=-1, high=1, size=(num_gen_data, num_features))
        # data_all[:, 0] = data_all[:, 0]*2 #multiply first feature by a constant
        data_all_labels = data_all[:, 0] + data_all[:, 0] ** 2 + data_all[:, 0] ** 3 + data_all[:, 0] ** 4
        data_all_labels = np.reshape(data_all_labels, newshape=(-1, 1))
        data_train = data_all[:num_train]
        data_train_labels = data_all_labels[:num_train]
        data_val = data_all[num_train:]
        data_val_labels = data_all_labels[num_train:]
    elif dataset == "poly4f":
        num_gen_data = num_train + num_val
        data_all = np.random.uniform(low=-1, high=1, size=(num_gen_data, num_features))
        # data_all[:, 0] = data_all[:, 0]*2 #multiply first feature by a constant
        data_all_labels = data_all[:, 0] + data_all[:, 1] ** 2 + data_all[:, 2] ** 3 + data_all[:, 3] ** 4
        data_all_labels = np.reshape(data_all_labels, newshape=(-1, 1))
        data_train = data_all[:num_train]
        data_train_labels = data_all_labels[:num_train]
        data_val = data_all[num_train:]
        data_val_labels = data_all_labels[num_train:]
    elif dataset == "inv2by2":
        num_gen_data = num_train + num_val
        K = np.array([[1, 2], [1, 2]])
        S = np.random.uniform(low=-1, high=1, size=(num_gen_data, 2, 2))
        I = np.matmul(K, S)
        data_all = np.reshape(I, newshape=(num_gen_data, 4))
        data_all_labels = S[:, 0, 0] #we will only predict the first output value of the matrix.
        data_all_labels = np.reshape(data_all_labels, newshape=(-1, 1))
        data_train = data_all[:num_train]
        data_train_labels = data_all_labels[:num_train]
        data_val = data_all[num_train:]
        data_val_labels = data_all_labels[num_train:]

    return data_train, data_train_labels, data_val, data_val_labels
