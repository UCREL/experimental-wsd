#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --output=log/out/deberta_v3_base_semantic_similarity_variable_negatives_%A_%a.log
#SBATCH --error=log/error/deberta_v3_base_semantic_similarity_variable_negatives_%A_%a.log
#SBATCH --array=0-3


declare -A learning_rate_mapper
learning_rate_mapper[0]="1e-5"
learning_rate_mapper[1]="5e-5"
learning_rate_mapper[2]="1e-6"
learning_rate_mapper[3]="5e-6"

srun python $(pwd)/training_runs/semantic_similarity/train_and_evaluate_token_similarity_variables_negatives.py fit \
--config $(pwd)/training_runs/semantic_similarity/variable_negatives_configs/base_config.yaml \
--config $(pwd)/training_runs/semantic_similarity/variable_negatives_configs/deberta_config.yaml \
"$@" --model.learning_rate ${learning_rate_mapper[${SLURM_ARRAY_TASK_ID}]}