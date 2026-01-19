#!/bin/bash

NUM_GPUS=8
COMMON_ARGS="--num_gpus $NUM_GPUS --batch_size 2"

EXP_NAMES=(
    # "lact_baseline_bs32_20k"
    # "lact_baseline_bs32_20k_ga_dot_product"
    # "lact_baseline_bs32_20k_mse"
    # "lact_baseline_bs32_20k_no_query_dot_product"
    # "lact_baseline_bs32_20k_only_w1_straight_qk"
    # "lact_baseline_bs32_20k_only_w1"
    "lact_baseline_bs32_20k_only_w1_no_momentum"
    # "lact_baseline_bs32_20k_only_w1_no_wn" # not finished
    "lact_baseline_bs32_20k_only_w1_straight_qk_no_lr1_no_momen"
    # "lact_baseline_bs32_20k_only_w1_straight_qk_no_lr1_no_lr1_no_muon"
    "lact_baseline_bs32_20k_only_w1_straight_qk_no_lr1_no_wn_muon_momen"
    # "lact_baseline_bs32_20k_only_w1_straight_qk_no_lr1_no_wn"
)

for EXP_NAME in "${EXP_NAMES[@]}"; do 
    ARGS="--model_path /lustre/fs12/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/users/ruilongl/flame/exp/$EXP_NAME $COMMON_ARGS"
    bash srun_sub.sh $NUM_GPUS eval_ppl_by_position nvr_torontoai_videogen $ARGS
done