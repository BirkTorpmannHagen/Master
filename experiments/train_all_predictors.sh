models=("Unet" "FPN" "TriUnet" "InductiveNet")

for model in ${models[*]} ; do
    export EXPERIMENT_MODEL=$model
    sbatch --array=[2-10%1] Experiments/enqueue_all.sbatch
done