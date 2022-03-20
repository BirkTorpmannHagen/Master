export EXPERIMENT_MODEL="DeepLab"
sbatch --array=[1-10%2] Experiments/enqueue_all.sbatch
export EXPERIMENT_MODEL="TriUnet"
sbatch --array=[5-10%1] Experiments/enqueue_all.sbatch
