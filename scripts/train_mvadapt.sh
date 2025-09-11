#!/bin/bash

source set_environment.sh

export DIRECT=0
export AGENTCONFIG="${WORK_DIR}/pretrained_models/leaderboard/tfpp_wp_all_0"
export UNCERTAINTY_THRESHOLD=0.33
export STOP_CONTROL=0

export VEHICLE="all"
export ROOT_DIR="${WORK_DIR}/dataset"
export BASE_MODEL=$AGENTCONFIG
export DEVICE="cuda:0"
export EPOCHS=100
export LR=0.0001
export BATCH_SIZE=512
export PROCESS_BATCH=64
export VERBOSE=true

export REMOVE_CRASHED=false
export REMOVE_IMPERFECT=false
# export MOVE_DUP_DIR="${WORK_DIR}/dataset_backup"

# export LOAD_DATA="${WORK_DIR}/dataset/mvadapt_dataset"
# export SAVE_DATA="None"
export LOAD_DATA="None"
export SAVE_DATA="${WORK_DIR}/dataset/mvadapt_dataset"

# export LOAD_MODEL="${WORK_DIR}/pretrained_models/mvadapt.pth"
# export SAVE_MODEL="None"
export LOAD_MODEL="None"
export SAVE_MODEL="${WORK_DIR}/pretrained_models/mvadapt.pth"


python ${WORK_DIR}/team_code_mvadapt/train_mvadapt.py \
--vehicle_ids=${VEHICLE} \
--root_dir=${ROOT_DIR} \
--base_model=${BASE_MODEL} \
--device=${DEVICE} \
--epochs=${EPOCHS} \
--lr=${LR} \
--batch_size=${BATCH_SIZE} \
--process_batch=${PROCESS_BATCH} \
--save_data=${SAVE_DATA} \
--save_model=${SAVE_MODEL} \
--load_data=${LOAD_DATA} \
--load_model=${LOAD_MODEL} \
--verbose=${VERBOSE} \
--remove_crashed=${REMOVE_CRASHED} \
--remove_imperfect=${REMOVE_IMPERFECT} \
--move_dup_dir=${MOVE_DUP_DIR}