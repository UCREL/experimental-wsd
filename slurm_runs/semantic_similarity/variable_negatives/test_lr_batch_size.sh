#!/bin/bash
#SBATCH --partition=a5000-6h
#SBATCH --output=log/out/semantic_similarity_variable_negatives_test_lr_batch_size.log
#SBATCH --error=log/error/semantic_similarity_variable_negatives_test_lr_batch_size.log
#SBATCH --ntasks-per-node=1


srun python $(pwd)/training_runs/semantic_similarity/get_token_similarity_variables_negatives_lr_batch_sizes.py \
--config training_runs/semantic_similarity/variable_negatives_configs/base_config.yaml \
"$@"