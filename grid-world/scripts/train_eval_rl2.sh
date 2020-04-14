#!/bin/bash
# This script reproduces Figure 4 & 5 in the paper
play_train="--env-name playground --tr_epi 10 --num-updates 8000"
play_eval="--env-name playground --tr_epi 20"

# Playground train (D1)
bash scripts/a_train_rlrl.sh $play_train ${@:1}

# Playground eval (D1~D4)
bash scripts/a_eval_rlrl.sh --level 1 --max_step 60 --ntasks 13 $play_eval ${@:1}
bash scripts/a_eval_rlrl.sh --level 2 --max_step 65 --ntasks 15 $play_eval ${@:1}
bash scripts/a_eval_rlrl.sh --level 3 --max_step 70 --ntasks 16 $play_eval ${@:1}
bash scripts/a_eval_rlrl.sh --level 4 --max_step 70 --ntasks 16 $play_eval ${@:1}

# Mining train
MINING_TRAIN="--env-name mining --max_step 70 --tr_epi 25"
bash scripts/a_train_rlrl.sh $MINING_TRAIN ${@:1}

# Mining eval set
MINING_EVAL="--env-name mining --max_step 70 --tr_epi 50"
bash scripts/a_eval_rlrl.sh $MINING_EVAL ${@:1}