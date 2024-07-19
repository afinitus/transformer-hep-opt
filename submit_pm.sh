#!/bin/bash

# Job name
#SBATCH --job-name=deepsets_preprocessed

# choose the GPU queue
#SBATCH --qos=regular
#SBATCH -C gpu
#SBATCH -A m3246

# request nodes
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -t 24:00:00


# note! this needs to match --trainer.devices!
#SBATCH --ntasks-per-node=4

# request enough memory
##SBATCH --mem=100G

# Change log names; %j gives job id, %x gives job name
#SBATCH --output=/pscratch/sd/n/nishank/humberto/slurms/slurm-%j.%x.out
# optional separate error output file
# activate environment
module load tensorflow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# run the training
echo "Running training script..."
python /pscratch/sd/n/nishank/humberto/transformer-hep/train_deepsetsatt.py -n 4
