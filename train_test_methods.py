#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 05:01:31 2022

@author: gjsaave
"""

import numpy as np
import sys
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append("../../")
import time


def train(train_loader, model, criterion, optimizer, batch_size, dataset, ml_task, reg_output_hook=None, l1_lambda=0):
    correct = 0
    total = 0
    acc = 0
    
    model.train()
    for i, (input, target) in enumerate(train_loader):     
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var, ml_task)

        if ml_task == "recon":
            input_var = torch.flatten(input_var, start_dim=1)
            target_var_correct_type = torch.flatten(target_var_correct_type, start_dim=1)

        # print(input_var)
        # print(input_var.shape)
        # sys.exit()

        optimizer.zero_grad()
        output = model(input_var.float())
        loss = criterion(output, target_var_correct_type)
        # print("------------------------------")
        # print(output)
        # print("target var ", target_var)
        # sys.exit()

        l1_penalty = 0
        if reg_output_hook is not None:
            for output in reg_output_hook:
                # print("output: ", output)
                l1_penalty += torch.norm(output, 1)
            l1_penalty *= l1_lambda

        # print("mse loss: ", loss)
        # print("l1 penalty: ", l1_penalty)
        loss = loss + l1_penalty

        loss.backward()
        optimizer.step()
        if reg_output_hook is not None:
            reg_output_hook.clear()
        # print(list(model.parameters())[0].grad)
        # sys.exit()

        if target.dim() == 1: #should be true for classification
            # print("Predictions may not be correct if target dimension is greater than 1. Exiting...")
            # sys.exit()
            pred = output.data.max(1)[1] #gets node index with max output. index corresponds to class prediction.
            correct += pred.eq(target.data).sum().item() #matches indices between target_var tensor and pred tensor.
            total += batch_size

    if ml_task == "class":
        acc = correct / total
        
    return acc, loss, model

def validation(val_loader, model, criterion, batch_size, dataset, ml_task, reg_output_hook=None, l1_lambda=0):
    correct = 0
    total = 0
    acc = 0

    model.eval()
    # if dropout:
    #     model.apply(apply_dropout)
        
    for i, (input, target) in enumerate(val_loader):     
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var, ml_task)

        if ml_task == "recon":
            input_var = torch.flatten(input_var, start_dim=1)
            target_var_correct_type = torch.flatten(target_var_correct_type, start_dim=1)

        with torch.no_grad():
            output = model(input_var.float())
            loss = criterion(output, target_var_correct_type)
            # print(output)
            # print(target_var_correct_type)
            # print("-----------------------------")

            l1_penalty = 0
            if reg_output_hook is not None:
                for output in reg_output_hook:
                    l1_penalty += torch.norm(output, 1)
                l1_penalty *= l1_lambda

            loss = loss + l1_penalty

            if reg_output_hook is not None:
                reg_output_hook.clear()

            if target.dim() == 1:  #should be true for classification
                # print("Predictions may not be correct if target dimension is greater than 1. Exiting...")
                # sys.exit()
                pred = output.data.max(1)[1] #gets node index with max output. index corresponds to class prediction.
                correct += pred.eq(target.data).sum().item() #matches indices between target_var tensor and pred tensor.
                total += batch_size

    if ml_task == "class":
        acc = correct / total
    #print("Test Accuracy: ", acc)
    return acc, loss


def get_nn_output(loader, model, criterion, batch_size, dataset, ml_task):
    correct = 0
    total = 0
    acc = 0
    outputs = []

    model.eval()
    # if dropout:
    #     model.apply(apply_dropout)

    for i, (input, target) in enumerate(loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var, ml_task)

        with torch.no_grad():
            output = model(input_var.float())
            loss = criterion(output, target_var_correct_type)
            # print(output)
            # print(target_var_correct_type)
            # print("-----------------------------")
            outputs.append(output.item())

            if target.dim() == 1:  # should be true for classification
                # print("Predictions may not be correct if target dimension is greater than 1. Exiting...")
                # sys.exit()
                pred = output.data.max(1)[1]  # gets node index with max output. index corresponds to class prediction.
                correct += pred.eq(
                    target.data).sum().item()  # matches indices between target_var tensor and pred tensor.
                total += batch_size

    if ml_task == "class":
        acc = correct / total
    # print("Test Accuracy: ", acc)
    return acc, loss, outputs


def get_nn_output_from_hook(loader, model, criterion, batch_size, dataset, ml_task, output_hook):
    correct = 0
    total = 0
    acc = 0
    outputs = []

    model.eval()
    # if dropout:
    #     model.apply(apply_dropout)

    for i, (input, target) in enumerate(loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var, ml_task)

        with torch.no_grad():
            output = model(input_var.float())
            loss = criterion(output, target_var_correct_type)
            # print(output)
            # print(target_var_correct_type)
            # print(output_hook)
            # print("-----------------------------")
            outputs.append(copy.copy(output_hook[0]))
            output_hook.clear()

            if target.dim() == 1:  # should be true for classification
                # print("Predictions may not be correct if target dimension is greater than 1. Exiting...")
                # sys.exit()
                pred = output.data.max(1)[1]  # gets node index with max output. index corresponds to class prediction.
                correct += pred.eq(
                    target.data).sum().item()  # matches indices between target_var tensor and pred tensor.
                total += batch_size

    if ml_task == "class":
        acc = correct / total
    # print("Test Accuracy: ", acc)
    return acc, loss, outputs

def get_target_indices(dataset, target):
    if dataset == "peaks":
        target_indices = target.data.max(1, keepdim=True).indices
    elif dataset == "mnist" or dataset == "cifar10":
        target_indices = target.data.unsqueeze(1)

    return target_indices


def get_correct_var_type(dataset, target_var, ml_task):
    if ml_task == "class":
        if dataset == "peaks":
            target_var_correct_type = target_var.float()
        elif dataset == "mnist" or dataset == "cifar10" or dataset == "peakspositive":
            target_var_correct_type = target_var.long()
        elif "bin" in dataset:
            target_var_correct_type = target_var.long()
    else:
        target_var_correct_type = target_var.float()

    return target_var_correct_type

def train_and_val_old(train_loader, val_loader, model, criterion, optimizer, batch_size, train_accs_all_iters, val_accs_all_iters, val_every_n_iters, full_idx, num_evals=1, dataset="peaks", device="cpu", val_iters_list=[]):
    train_correct = 0
    train_total = 0
    train_acc = 0
    
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var_correct_type = get_correct_var_type(dataset, target_var)
        
        for j in range(num_evals):
            optimizer.zero_grad()
            train_output = model(input_var.float())
            train_loss = criterion(train_output, target_var_correct_type, None, None, None, None)
            train_loss.backward()
            optimizer.step()
        #print("Train Loss: ", loss.item())
        
        train_pred = train_output.data.max(1, keepdim=True)[1]
        target_indices = get_target_indices(dataset, target)
        train_correct += train_pred.eq(target_indices).sum().item()
        train_total += input_var.shape[0]
        train_acc = train_correct / train_total
        #print("Train Accuracy: ", acc)
        train_accs_all_iters.append(train_acc)

        #Evaluate entire validation set every n iters or when training epoch is over
        if full_idx % val_every_n_iters == 0 or i == len(train_loader)-1:
            val_iters_list.append(full_idx)
            val_correct = 0
            val_total = 0
            val_acc = 0

            model.eval()

            for i, (input, target) in enumerate(val_loader):
                input, target = input.to(device), target.to(device)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)
                target_var_correct_type = get_correct_var_type(dataset, target_var)     

                with torch.no_grad():
                    val_output = model(input_var.float())
                    val_loss = criterion(val_output, target_var_correct_type, None, None, None, None)

                    val_pred = val_output.data.max(1, keepdim=True)[1]
                    target_indices = get_target_indices(dataset, target)
                    val_correct += val_pred.eq(target_indices).sum().item()
                    val_total += input_var.shape[0]
                    val_acc = val_correct / val_total

            val_accs_all_iters.append(val_acc)
            model.train()

        full_idx += 1
        
    return train_acc, train_loss, val_acc, val_loss, train_accs_all_iters, val_accs_all_iters, model, full_idx, val_iters_list


def get_data_indices(i, dataset_size, batch_size, levels):
    data_indices = []
    step = int(batch_size/(2*levels + 1))
    total = batch_size

    #If the batch is too small we just use the same indices for whole iteration og smgdrop
    if (i+1)*batch_size > dataset_size:
        for k in range(0, total, step):
            data_indices.append((0, -1))

    else:
        last_k = 0
        for k in range(0, total, step):
            data_indices.append((k, k+step))
            last_k = k
            
    return data_indices

