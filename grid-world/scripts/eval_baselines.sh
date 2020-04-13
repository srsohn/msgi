# This script reproduces Figure 4 & 5 in the paper
# Run Random & GRProp+Oracle agents
play_train="--env-name playground --tr_epi 10 --num-updates 8000"
play_eval="--env-name playground --tr_epi 20"

# Playground eval (D1~D4)
bash scripts/a_eval_policy.sh --algo grprop --level 1 --max_step 60 --ntasks 13 ${@:1}
bash scripts/a_eval_policy.sh --algo grprop --level 2 --max_step 65 --ntasks 15 ${@:1}
bash scripts/a_eval_policy.sh --algo grprop --level 3 --max_step 70 --ntasks 16 ${@:1}
bash scripts/a_eval_policy.sh --algo grprop --level 4 --max_step 70 --ntasks 16 ${@:1}

# Mining eval set
MINING_EVAL="--env-name mining --max_step 70 --tr_epi 50"
bash scripts/a_eval_policy.sh --algo random $MINING_EVAL ${@:1}

# Playground eval (D1~D4)
bash scripts/a_eval_policy.sh --algo random --level 1 --max_step 60 --ntasks 13 ${@:1}
bash scripts/a_eval_policy.sh --algo random --level 2 --max_step 65 --ntasks 15 ${@:1}
bash scripts/a_eval_policy.sh --algo random --level 3 --max_step 70 --ntasks 16 ${@:1}
bash scripts/a_eval_policy.sh --algo random --level 4 --max_step 70 --ntasks 16 ${@:1}

# Mining eval set
MINING_EVAL="--env-name mining --max_step 70 --tr_epi 50"
bash scripts/a_eval_policy.sh --algo grprop $MINING_EVAL ${@:1}