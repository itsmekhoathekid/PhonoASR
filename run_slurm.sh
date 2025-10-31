#!/bin/bash
#SBATCH --job-name= PhonoASR_TASA
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=mps:1 # dùng 1 slot MPS
#SBATCH --output=log_test.txt

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
srun python3 /datastore/npl/Speech2Text/PhonoASR/train.py --config /datastore/npl/Speech2Text/PhonoASR/configs/TASA-config.yaml

echo quit | nvidia-cuda-mps-control
