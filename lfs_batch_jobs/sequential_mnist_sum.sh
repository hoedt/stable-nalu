#!/bin/bash
experiment_name='sequential_mnist_sum'

for seed in {0..10}
do
    bsub -q compute -n 8 -W 20:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation sum --layer-type NAC \
        --interpolation-length 10 --extrapolation-short-length 100 --extrapolation-long-length 1000 \
        --seed ${seed} --max-epochs 50000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 20:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation sum --layer-type NALU \
        --interpolation-length 10 --extrapolation-short-length 100 --extrapolation-long-length 1000 \
        --seed ${seed} --max-epochs 50000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data

    bsub -q compute -n 8 -W 20:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name} -e /work3/$USER/logs/${experiment_name} -R "span[hosts=1]" -R "rusage[mem=10GB]" ./python_lfs_job.sh \
        experiments/sequential_mnist.py \
        --operation sum --layer-type ReRegualizedLinearNAC \
        --interpolation-length 10 --extrapolation-short-length 100 --extrapolation-long-length 1000 \
        --seed ${seed} --max-epochs 50000 --verbose \
        --name-prefix ${experiment_name} --remove-existing-data
done