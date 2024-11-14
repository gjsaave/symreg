#!/bin/bash


dataset="rk1"

loss_method="mse"
num_val=100

if [[ $dataset == "rk1" ]] || [[ $dataset == "rk1t" ]]
then
  num_features=1
  num_output_nodes=1
fi

CURFOLDER=$(basename "$PWD")

savepath="/users/garysaavedra/exp_output/symreg/${CURFOLDER}"

#for num train = 10
#nngen_model_path="/Users/garysaavedra/exp_output/symreg/exp00005_symreg_ensemble_nn/sd30_bs8_mm0.0_wd0.0_nl3_lr0.01_dr0.0_mtnonlinear_nh4_no1_omadam_dsrk1_nc_lmmse_nt10_nv100_upall_tlNone_ep10_ntb8"

nngen_model_type="random100"
nngen_model_path="/Users/garysaavedra/exp_output/symreg/exp00002_source_nn_hyper_grid_search/"

#for num train = 50
#nngen_model_path="/Users/garysaavedra/exp_output/symreg/exp00002_source_nn_hyper_grid_search/sd30_bs8_mm0.0_wd0.0_nl2_lr0.1_dr0.0_mtnonlinear_nh16_no1_omadam_dsrk1_nc_lmmse_nt50_nv100_upall_tlNone_ep10"

#for num train = 100
#nngen_model_path="/Users/garysaavedra/exp_output/symreg/exp00002_source_nn_hyper_grid_search/sd32_bs8_mm0.0_wd0.0_nl3_lr0.1_dr0.0_mtnonlinear_nh16_no1_omadam_dsrk1_nc_lmmse_nt100_nv100_upall_tlNone_ep10"

for seed in 30 31 32 33 34; do
for num_train in 10; do
for population in 750; do
for p_crossover in 0.5; do
for p_subtree_mutation in 0.1; do
for p_point_mutation in 0.1; do
for generations in 50; do
for nngen in 1000; do

python run_symreg_test.py --generations=$generations --savepath=$savepath --seed=$seed --num_features=$num_features --dataset=$dataset --loss_method=$loss_method --num_train=$num_train --num_val=$num_val --population=$population --p_crossover=$p_crossover --p_subtree_mutation=$p_subtree_mutation --p_point_mutation=$p_point_mutation --generations=$generations --nngen=$nngen --nngen_model_path=$nngen_model_path --nngen_model_type=$nngen_model_type

done
done
done
done
done
done
done
done
