#!/bin/bash
#SBATCH --job-name=lact_baseline_bs1_debug
#SBATCH --output=exp/lact_baseline_bs1_debug/%x_%j.out
#SBATCH --error=exp/lact_baseline_bs1_debug/%x_%j.err
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

echo "=============================================="
echo "LaCT Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: 1x H100"
echo "Start time: $(date)"
echo "=============================================="

# Load modules (Trillium GPU cluster)
module load StdEnv/2023 gcc/13.3
module load cuda/12.6
module load python/3.11.5

# Environment variables
export OMP_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Offline mode (compute nodes have no internet)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# uv experimental features
export UV_PREVIEW_FEATURES=extra-build-dependencies

# Trillium-specific: scratch for output (home/project are read-only on compute nodes)
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export TRITON_CACHE_DIR="$SCRATCH/.triton/cache"
export HF_DATASETS_CACHE="$SCRATCH/datasets/fineweb-edu/sample-100BT"
export WANDB_DIR="$SCRATCH/.cache/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"

# check allocated resources
srun nvidia-smi

echo
echo "Starting training..."
echo

# Run the training job
export NGPU=1
export NNODE=1
export WANDB_PROJECT="lact"
export WANDB_NAME="lact_baseline_bs1_debug"

srun bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder $SCRATCH/flame/exp/$WANDB_NAME \
  --model.config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 32768 \
  --training.context_len 32768 \
  --training.gradient_accumulation_steps 1 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 1 \
  --training.steps 40960 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-100BT \
  --training.dataset_split train \
  --training.num_workers 2 \
  --training.prefetch_factor 1 \
  --training.seed 42 \
  --checkpoint.interval 4096 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1 \
  --profiling.profile_freq 2000

echo
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="

exit 0
