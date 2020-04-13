#!/bin/bash
msgi_meta="--mode meta_eval --infer --algo a2c --method SGI"
common_param="--flat_ldim 512 --gru_ldim 512 --nworker 8"

python main.py $msgi_meta $common_param ${@:1}