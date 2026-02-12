export WANDB_MODE=offline

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
