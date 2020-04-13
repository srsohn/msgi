# This script reproduces Figure 4 & 5 in the paper
# Run MSGI-Random agents
play_train="--env-name playground --tr_epi 10 --num-updates 8000"
play_eval="--env-name playground --tr_epi 20"

# Playground eval (D1~D4)
bash scripts/a_eval_msgi_random.sh --level 1 --max_step 60 --ntasks 13 $play_eval  ${@:1}
bash scripts/a_eval_msgi_random.sh --level 2 --max_step 65 --ntasks 15 $play_eval  ${@:1}
bash scripts/a_eval_msgi_random.sh --level 3 --max_step 70 --ntasks 16 $play_eval  ${@:1}
bash scripts/a_eval_msgi_random.sh --level 4 --max_step 70 --ntasks 16 $play_eval  ${@:1}

# Mining eval set
MINING_EVAL="--env-name mining --max_step 70 --tr_epi 50"
bash scripts/a_eval_msgi_random.sh $MINING_EVAL ${@:1}