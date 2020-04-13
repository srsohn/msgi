# This script reproduces Figure 4 & 5 in the paper
play_train="--env-name playground --tr_epi 10 --num-updates 8000"
play_eval="--env-name playground --tr_epi 20"

# Playground train (D1)
bash scripts/a_train_msgi_meta.sh --exp-id 30 --seed 1 $play_train  ${@:1}

# Playground eval (D1~D4)
load="--load_dir SGI_a2c_UCB_lv1_epi10_301"
bash scripts/a_eval_msgi_meta.sh --level 1 --max_step 60 --ntasks 13 $play_eval $load ${@:1}
bash scripts/a_eval_msgi_meta.sh --level 2 --max_step 65 --ntasks 15 $play_eval $load ${@:1}
bash scripts/a_eval_msgi_meta.sh --level 3 --max_step 70 --ntasks 16 $play_eval $load ${@:1}
bash scripts/a_eval_msgi_meta.sh --level 4 --max_step 70 --ntasks 16 $play_eval $load ${@:1}

# Mining train
MINING_TRAIN="--env-name mining --max_step 70 --tr_epi 25"
bash scripts/a_train_msgi_meta.sh --exp-id 30 --seed 1 $MINING_TRAIN ${@:1}

# Mining eval set
load="--load_dir Mine_SGI_a2c_UCB_epi25_301"
MINING_EVAL="--env-name mining --max_step 70 --tr_epi 50"
bash scripts/a_eval_msgi_meta.sh $MINING_EVAL ${@:1}