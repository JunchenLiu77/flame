#!/bin/bash

NUM_GPUS=8
COMMON_ARGS="--task niah_single_2 --num_gpus $NUM_GPUS --limit 500"

CHECKPOINT_STEPS=(10000)
SEQ_LENGTHS=(1024 2048 4096 8192)

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do
  for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    ARGS="--checkpoint exp/lact_baseline_bs32_20k/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS
  done
done

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do
  for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    ARGS="--checkpoint exp/lact_baseline_bs32_20k_ga_dot_product/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_ga_dot_product.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS
  done
done

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do
  for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    ARGS="--checkpoint exp/lact_baseline_bs32_20k_mse/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_mse.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS
  done
done

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do
  for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    ARGS="--checkpoint exp/lact_baseline_bs32_20k_no_query_dot_product/checkpoint/step-$CHECKPOINT_STEP --config configs/lact_baseline_bs32_20k_no_query_dot_product.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS
  done
done

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do
  for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1_straight_qk/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1_straight_qk.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS
  done
done

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do
  for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    ARGS="--checkpoint exp/lact_baseline_bs32_20k_only_w1/checkpoint/step-$CHECKPOINT_STEP --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_only_w1.json --seq_length $SEQ_LENGTH $COMMON_ARGS"
    bash srun_sub.sh $NUM_GPUS eval_ruler nvr_torontoai_videogen $ARGS
  done
done
