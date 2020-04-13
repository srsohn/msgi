#!/bin/bash
hrl="--mode meta_eval --train --algo a2c --method baseline --nworker 1"
common_param="--use-gae --tau 0.9 --flat_ldim 512 --gru_ldim 512 --nworker 8"
hyper_param="--lr 0.001 --lr_decay 0.999 --v-coef 0.12 --rho-v-st 0.03"

python main.py $hrl $common_param ${@:1}