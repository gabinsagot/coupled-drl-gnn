#!/bin/bash
#
#SBATCH --job-name=gnn_pred
#SBATCH --output=pred.log
#
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=node-99
#SBATCH --mail-type=FAIL
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1g.5gb:1

# Set environment variable for OpenMP
export OMP_PROC_BIND=true

# Set wand offline
export WANDB_MODE=offline

# Execute MPI run
python -m graphphysics.predict \
            --predict_parameters_path=predict_config/airfoil.json \
            --model_path=checkpoints/model.ckpt \
            --no_edge_feature
