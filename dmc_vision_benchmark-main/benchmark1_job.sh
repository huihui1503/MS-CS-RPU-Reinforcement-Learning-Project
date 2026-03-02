#!/bin/bash
#SBATCH --job-name=dmc-benchmark-1      # Name of your job
#SBATCH --output=logs/%x_%j.out        # Output file: logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err         # Error file: logs/jobname_jobid.err
#SBATCH --partition=informatik-mind    # Partition to submit to (e.g., gpu, long, short)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=490G
#SBATCH --time=20-00:00:00
#SBATCH --mail-type=END,FAIL           # Notifications


# Start GPU logging in background
# nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
#   --format=csv -l 10 > logs/%x_%j_gpu_usage.log &

# GPU_LOG_PID=$!

module load ffmpeg

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate jax-gpu_v2

# Run 
unset DISPLAY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

START=$(date +%s)
export LD_LIBRARY_PATH=/home/cor54gyp/.conda/envs/jax-gpu/lib

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
xvfb-run -a python example/benchmark1.py
# xvfb-run -a python example/example.py

END=$(date +%s)
echo "Runtime: $((END - START)) seconds"

conda deactivate

# Stop GPU logging when training finishes
# kill $GPU_LOG_PID