#!/bin/bash


dataset="poly4f"

loss_method="mse"
num_val=100

if [[ $dataset == "rk1" ]] || [[ $dataset == "rk1t" ]]
then
  num_features=1
  num_output_nodes=1
elif [[ $dataset == "poly4f" ]]
then
  num_features=4
  num_output_nodes=1
fi

CURFOLDER=$(basename "$PWD")

savepath="/users/garysaavedra/exp_output/symreg/${CURFOLDER}"

for seed in 30 31 32 33 34; do
for num_train in 10000; do
for population in 750; do
for p_crossover in 0.7; do
for p_subtree_mutation in 0.15; do
for p_point_mutation in 0.1; do
for generations in 50; do

python run_symreg_test.py --generations=$generations --savepath=$savepath --seed=$seed --num_features=$num_features --dataset=$dataset --loss_method=$loss_method --num_train=$num_train --num_val=$num_val --population=$population --p_crossover=$p_crossover --p_subtree_mutation=$p_subtree_mutation --p_point_mutation=$p_point_mutation --generations=$generations

done
done
done
done
done
done
done
