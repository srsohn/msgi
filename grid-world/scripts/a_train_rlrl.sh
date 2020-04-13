#!/bin/bash
rlrl="--mode meta_train --algo a2c --train --method rlrl"
common_param="--use-gae --tau 0.9 --flat_ldim 512 --gru_ldim 512 --nworker 24"
hyper_param="--lr 0.001 --lr_decay 0.999 --v-coef 0.005 --rho-v-st 1.0"

python main.py $rlrl $common_param $hyper_param ${@:1}