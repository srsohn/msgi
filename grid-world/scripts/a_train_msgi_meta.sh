#!/bin/bash
msgi_meta="--mode meta_train --infer --algo a2c --train --method SGI"
common_param="--use-gae --tau 0.9 --flat_ldim 512 --gru_ldim 512 --nworker 24"
hyper_param="--lr 0.002 --lr_decay 0.999 --v-coef 0.12 --rho-v-st 0.1 --bonus 2"

python main.py $msgi_meta $common_param $hyper_param ${@:1}