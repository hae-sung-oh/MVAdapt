#!/bin/bash

source set_environment.sh

export DIRECT=0
export AGENTCONFIG="${WORK_DIR}/pretrained_models/leaderboard/tfpp_wp_all_0"
export UNCERTAINTY_THRESHOLD=0.33
export STOP_CONTROL=0

export VEHICLE=6 # carlamotors.carlacola
export ROOT_DIR="${WORK_DIR}/dataset"
export BASE_MODEL=$AGENTCONFIG
export DEVICE="cuda:0"
export EPOCHS=20
export LR=0.0001
export BATCH_SIZE=512
export PROCESS_BATCH=64
export VERBOSE=true

# export LOAD_DATA="${WORK_DIR}/dataset/mvadapt_finetune_dataset"
# export SAVE_DATA="None"
export LOAD_DATA="None"
export SAVE_DATA="${WORK_DIR}/dataset/mvadapt_finetune_dataset"

export PRETRAINED_MODEL="${WORK_DIR}/pretrained_models/mvadapt.pth"
export SAVE_FINETUNED_MODEL="${WORK_DIR}/pretrained_models/mvadapt_finetuned.pth"


python ${WORK_DIR}/team_code_mvadapt/finetune_mvadapt.py \
--unseen_vehicle_id=${VEHICLE} \
--root_dir=${ROOT_DIR} \
--base_model=${BASE_MODEL} \
--device=${DEVICE} \
--finetune_epochs=${EPOCHS} \
--finetune_lr=${LR} \
--batch_size=${BATCH_SIZE} \
--process_batch=${PROCESS_BATCH} \
--save_data=${SAVE_DATA} \
--load_data=${LOAD_DATA} \
--pretrained_model=${PRETRAINED_MODEL} \
--save_finetuned_model=${SAVE_FINETUNED_MODEL} 