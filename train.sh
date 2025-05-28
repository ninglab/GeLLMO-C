#!/bin/bash

# Usage: train.sh <model> <data_dir> <expt_dir> <num_epochs> <max_steps> <tasks> <batch_size> <resume_from_checkpoint>
# You may need about 16 GB RAM

base_model=$1
data_path=$2
expt_dir=$3
num_epochs=$4
max_steps=$5
tasks=$6
opt_type="thresh"
task_column="property_comb"
batch_size=$7
resume_from_checkpoint=$8

model_lower=$(echo $base_model | tr '[:upper:]' '[:lower:]')
lora_target_modules="['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head']"

#if resume_from_checkpoint is not None, then the model will be loaded from the checkpoint, else set to False
if [[ $resume_from_checkpoint == "" ]]; then
   resume_from_checkpoint=False
fi

if [[ $max_steps == 0 ]]; then
   python trainer.py \
      --data_path $data_path \
      --base_model $base_model \
      --lora_target_modules $lora_target_modules \
      --output_dir $expt_dir \
      --opt_type $opt_type \
      --resume_from_checkpoint $resume_from_checkpoint \
      --tasks $tasks --task_column $task_column \
      --batch_size $batch_size
else
   python trainer.py \
      --data_path $data_path \
      --base_model $base_model \
      --lora_target_modules $lora_target_modules \
      --output_dir $expt_dir \
      --num_epochs $num_epochs \
      --opt_type $opt_type \
      --resume_from_checkpoint $resume_from_checkpoint \
      --tasks $tasks --task_column $task_column \
      --batch_size $batch_size --max_steps $max_steps
fi