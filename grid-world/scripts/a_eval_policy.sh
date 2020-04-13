#!/bin/bash
common_param="--nworker 8"
specific="--mode eval --method baseline"
python main.py $common_param $specific ${@:1}