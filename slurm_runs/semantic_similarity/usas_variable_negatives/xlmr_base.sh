#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --output=log/out/xlmr_base_usas_semantic_similarity_variable_negatives.log
#SBATCH --error=log/error/xlmr_base_usas_semantic_similarity_variable_negatives.log

srun python $(pwd)/training_runs/usas_semantic_similarity/train_and_evaluate_token_similarity_variables_negatives.py fit \
--config $(pwd)/training_runs/usas_semantic_similarity/variable_negatives_configs/base_config.yaml \
--config $(pwd)/training_runs/usas_semantic_similarity/variable_negatives_configs/xlmr_base.yaml \
"$@"