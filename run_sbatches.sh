#!/bin/bash

msg_shapes=(16 32 64)
algos=('coma' 'maa2c' 'maddpg' 'mappo' 'qmix')
for msg_shape in "${msg_shapes[@]}"; do
    for algo in "${algos[@]}"; do
        sbatch slurm.sh $algo $msg_shape
    done
done