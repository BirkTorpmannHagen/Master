MODELS=["DeepLab" "UNet" "DivergentNet", "DDANet", ""]
for i in {0..10} ; do
    sbatch enqueue_ex3.sbatch $i
done