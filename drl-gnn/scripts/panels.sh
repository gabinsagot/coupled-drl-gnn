#!/bin/bash
#
#SBATCH --job-name=panelsPBO
#SBATCH --output=panels.log
#
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=24:00:00
#

# OpenMPI module
module load openmpi/4.1.1

# Set environment variable for tf_keras
export TF_USE_LEGACY_KERAS=True

# Suppress TensorFlow info and warning messages
export TF_CPP_MIN_LOG_LEVEL=2  
export TF_CPP_MIN_VLOG_LEVEL=0
export ABSL_LOG=0

# pbo 
pbo optim_config/panels.json 2> panels.err
