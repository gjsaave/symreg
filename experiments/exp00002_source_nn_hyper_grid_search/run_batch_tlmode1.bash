#!/bin/bash

epochs=10

drop_rate=0 #Also controls the drop rate of top level of mgdrop
debug=False

momentum=0
weight_decay=0
model_type="load"
dataset="rk1t"
update_str="final"
tl_mode="final"

loss_method="mse"
ml_task="reg"
opt_method="adam"
num_val=100

batch_size=8

if [[ $dataset == "rk1" ]] || [[ $dataset == "rk1t" ]]
then
  num_features=1
  num_output_nodes=1
fi

CURFOLDER=$(basename "$PWD")

savepath="/users/garysaavedra/exp_output/symreg/${CURFOLDER}"
#model_path="/Users/garysaavedra/exp_output/symreg/exp00001_transfer_learn/sd30_bs8_mm0.0_wd0.0_nl2_lr0.1_dr0.0_mtnonlinear_nh8_no1_omadam_dsrk1_nc_lmmse_ng63_upall_tlNone/output/model.pt"
model_path=None

#don't need these for transfer learning
num_layers=None
num_hidden_nodes=None

#TODO Make list of model paths. Make list of num_train_source Each entry is the experiment with the best performing model. Then have index for num_train_source and read in num points this way

for seed in 30 31 32 33 34; do
for lr in 0.1 0.01; do
for num_train_source in 10 20 30 40 50 60 70 80 90 100; do
for num_train in 10 20 30 40 50 60 70 80 90 100; do

python train_nn.py --batch_size=$batch_size --epochs=$epochs --lr=$lr --datapath=$datapath --savepath=$savepath --num_hidden_nodes=$num_hidden_nodes --momentum=$momentum --weight_decay=$weight_decay --num_layers=$num_layers --drop_rate=$drop_rate --seed=$seed --num_features=$num_features --num_output_nodes=$num_output_nodes --model_type=$model_type  --opt_method=$opt_method --dataset=$dataset --loss_method=$loss_method --debug=$debug --ml_task=$ml_task --model_path=$model_path --update_str=$update_str --tl_mode=$tl_mode --num_train=$num_train --num_val=$num_val

done
done
done
done

