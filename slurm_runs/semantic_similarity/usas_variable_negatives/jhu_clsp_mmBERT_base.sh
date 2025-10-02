#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --output=log/out/jhu_clsp_mmbert_base_usas_semantic_similarity_variable_negatives.log
#SBATCH --error=log/error/jhu_clsp_mmbert_base_usas_semantic_similarity_variable_negatives.log

srun python $(pwd)/training_runs/usas_semantic_similarity/train_and_evaluate_token_similarity_variables_negatives.py fit \
--config $(pwd)/training_runs/usas_semantic_similarity/variable_negatives_configs/base_config.yaml \
--config $(pwd)/training_runs/usas_semantic_similarity/variable_negatives_configs/jhu_clsp_mmBERT_base.yaml \
--model.learning_rate 1e-5 --data.dataset_folder_name usas_semantic_similarity_variable_nagative_jhu_clsp_mmBERT_base_z99_filtered