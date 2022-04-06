
models=("DeepLab" "Unet" "FPN" "TriUnet" "InductiveNet")
for model in ${models[*]} ; do
    export EXPERIMENT_MODEL=$model
    sbatch --array=[8-10%1] experiments/enqueue_all.sbatch
done