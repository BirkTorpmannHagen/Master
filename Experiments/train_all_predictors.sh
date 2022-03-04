models=("DeepLab" "Unet" "FPN" "TriUnet" "InductiveNet")

for model in ${models[*]} ; do
    export EXPERIMENT_MODEL=$model
    sbatch --array=[1-10] Experiments/enqueue_all.sbatch
done