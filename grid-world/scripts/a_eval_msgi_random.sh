#!/bin/bash
msgi_random="--mode meta_eval --infer --algo random --method SGI"
common_param="--flat_ldim 512 --gru_ldim 512 --nworker 8"
python main.py $msgi_random $common_param ${@:1}