#!/bin/bash

CHECKPOINT_STEPS=(20000)
SEQ_LENGTHS=(1024 2048 4096 8192)

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do
  for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    if [ $SEQ_LENGTH -eq 1024 ]; then
      NUM_GPUS=1
    elif [ $SEQ_LENGTH -eq 2048 ]; then
      NUM_GPUS=2
    elif [ $SEQ_LENGTH -eq 4096 ]; then
      NUM_GPUS=4
    else
      NUM_GPUS=8
    fi

    COMMON_ARGS="--task niah_single_2 --num_gpus $NUM_GPUS --limit 500"

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_ga_dot_product/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_ga_dot_product.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_mse/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_mse.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_no_query_dot_product/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_no_query_dot_product.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1_straight_qk/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1_straight_qk.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1_no_momentum/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1_no_momentum.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1_no_wn/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1_no_wn.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1_straight_qk_no_lr1_no_momen/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1_straight_qk_no_lr1_no_momen.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1_straight_qk_no_lr1_no_lr1_no_muon/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1_straight_qk_no_lr1_no_muon.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1_straight_qk_no_lr1_no_wn_muon_momen/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1_straight_qk_no_lr1_no_wn_muon_momen.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

    # ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1_straight_qk_no_lr1_no_wn/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1_straight_qk_no_lr1_no_wn.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    # bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS

  done
done