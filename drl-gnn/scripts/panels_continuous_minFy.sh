#!/bin/bash
#
#SBATCH --job-name=PBO_c_Fy
#SBATCH --output=panels_continuous_minFy.log
#
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --time=168:00:00
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=node-99
#SBATCH --gres=gpu:1g.5gb:1

# OpenMPI module
module load openmpi/4.1.1

# Set environment variable for OpenMP
export OMP_PROC_BIND=true

# Set wand offline
export WANDB_MODE=offline

# Set environment variable for tf_keras
export TF_USE_LEGACY_KERAS=True

# Suppress TensorFlow info and warning messages
export TF_CPP_MIN_LOG_LEVEL=2  
export TF_CPP_MIN_VLOG_LEVEL=0
export ABSL_LOG=0

# pbo 
pbo optim_config/panels_continuous_minFy.json 2> panels_continuous_minFy.err
