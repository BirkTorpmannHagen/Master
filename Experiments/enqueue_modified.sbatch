#!/bin/bash
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -c 4 # number of cores
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

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=4
fi
export OMP_NUM_THREADS=$omp_threads        # OpenMP, Numpy
export MKL_NUM_THREADS=$omp_threads   # Intel MKL
export NUMEXPR_NUM_THREADS=$omp_threads      # Python3 Multiproc
# export OPENBLAS_NUM_THREADS=2     # Using OpenBLAS?
# export VECLIB_MAXIMUM_THREADS=2    # Accelware Vector Lib

export PYTHONPATH=$PWD
MODEL="DeepLab"
echo "Starting jobs"
srun python Experiments/train_predictors_for_model.py 1 $MODEL &
srun python Experiments/train_predictors_for_model.py 2 $MODEL &
srun python Experiments/train_predictors_for_model.py 3 $MODEL &
srun python Experiments/train_predictors_for_model.py 4 $MODEL &
srun python Experiments/train_predictors_for_model.py 5 $MODEL &
srun python Experiments/train_predictors_for_model.py 6 $MODEL &
srun python Experiments/train_predictors_for_model.py 7 $MODEL &
srun python Experiments/train_predictors_for_model.py 8 $MODEL &
srun python Experiments/train_predictors_for_model.py 9 $MODEL &
srun python Experiments/train_predictors_for_model.py 10 $MODEL &
echo "Jobs started"
wait