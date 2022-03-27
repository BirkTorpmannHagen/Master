models=("DeepLab" "Unet" "FPN" "TriUnet" "InductiveNet")

for model in ${models[*]} ; do
    export EXPERIMENT_MODEL=$model
    sbatch --array=[1-5%1] experiments/enqueue_all.sbatch
done