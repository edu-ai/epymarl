#!/bin/bash

#SBATCH --job-name=epymarl
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=80G                                     
#SBATCH --time=120:00:00

config=$1
env_name=${2:-MMM}
env_config=${3:-sc2}

cd /home/y/yzhilong/epymarl
source ../python_envs/epymarl38/bin/activate
echo "message_shape: "$message_shape
python3 src/main.py --config=$config --env-config=$env_config with env_args.map_name=$env_name allow_communications=false seed=1
