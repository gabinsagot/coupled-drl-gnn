#!/bin/bash
#
#SBATCH --job-name=gnn_arfl_Re3
#SBATCH --output=out.log
#
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=node-99
#SBATCH --gres=gpu:2g.10gb:1

# Set up PATHs (if not already set in ~/.bashrc)
export DGLDEFAULTDIR="/scratch-big/gsagot/.dgl"
export MPLCONFIGDIR="/scratch-big/gsagot/.config/matplotlib"

# Set wand offline
export WANDB_MODE=offline

# Execute MPI run
python -m graphphysics.train \
            --project_name=airfoilRe1000 \
            --training_parameters_path=training_config/airfoil.json \
            --num_epochs=20 \
            --init_lr=0.0001 \
            --batch_size=2 \
            --warmup=1500 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_name=model \
            --no_edge_feature
