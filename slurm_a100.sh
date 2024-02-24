#!/bin/bash

#SBATCH --job-name=epymarl
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=65G                                     
#SBATCH --time=120:00:00

config=$1
message_shape=${2:-32}
env_name=${3:-MMM}
env_config=${4:-sc2}

cd /home/y/yzhilong/epymarl
source ../python_envs/epymarl38/bin/activate
echo "message_shape: "$message_shape
python3 src/main.py --config=$config --env-config=$env_config with env_args.map_name=$env_name message_shape=$message_shape
