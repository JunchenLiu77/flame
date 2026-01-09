#!/bin/bash
#SBATCH --job-name=lact_baseline_bs32_20k
#SBATCH --output=exp/lact_baseline_bs32_20k/%x_%j.out
#SBATCH --error=exp/lact_baseline_bs32_20k/%x_%j.err
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --account=aip-fsanja
#SBATCH --exclude=kn122


echo "=============================================="
echo "LaCT Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: 8x H100"
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

# uv experimental features
export UV_PREVIEW_FEATURES=extra-build-dependencies
export HF_DATASETS_CACHE="/datasets/DL3DV-DSO/fineweb-edu/sample-100BT"

# check allocated resources
srun nvidia-smi

echo
echo "Starting training..."
echo

# Run the training job
export NGPU=8
export NNODE=1
export WANDB_PROJECT="lact"
export WANDB_NAME="lact_baseline_bs32_20k"

srun bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/$WANDB_NAME \
  --model.config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 4 \
  --training.seq_len 32768 \
  --training.context_len 32768 \
  --training.gradient_accumulation_steps 1 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 1 \
  --training.steps 20000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-100BT \
  --training.dataset_split train \
  --training.num_workers 2 \
  --training.prefetch_factor 1 \
  --training.seed 42 \
  --checkpoint.interval 1000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 4 \
  --metrics.log_freq 1 \
  --metrics.enable_wandb  \
  --profiling.profile_freq 5000

echo
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="

exit 0
