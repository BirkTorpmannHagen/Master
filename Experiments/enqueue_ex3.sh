#!/bin/bash
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -c 8 # number of cores
#SBATCH -w g001
#SBATCH --gres=gpu:1
#       #SBATCH --mem 128G # memory pool for all cores   # Removed due to bug in Slurm 20.02.5
#SBATCH -t 4-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

ulimit -s 10240

module purge
module load slurm/20.02.7
module load cuda11.0/blas/11.0.3
module load cuda11.0/fft/11.0.3
module load cuda11.0/nsight/11.0.3
module load cuda11.0/profiler/11.0.3
module load cuda11.0/toolkit/11.0.3

srun python train_augmented_pipeline.py