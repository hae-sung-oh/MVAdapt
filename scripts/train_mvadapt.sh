#!/bin/bash

source set_environment.sh

export DIRECT=0
export AGENTCONFIG="${WORK_DIR}/pretrained_models/leaderboard/tfpp_wp_all_0"
export UNCERTAINTY_THRESHOLD=0.33
export STOP_CONTROL=0

export VEHICLE="[1,2,6,7,11,20,24]"
export ROOT_DIR="${WORK_DIR}/dataset"
export BASE_MODEL=$AGENTCONFIG
export DIM0=32
export DIM1=32
export DIM2=64
export DIM3=64
export DEVICE="cuda:0"
export PHYSICS_DIM=18
export MAX_GEAR_NUM=24
export GEAR_DIM=4
export EPOCHS=1000
export LR=0.00001
export BATCH_SIZE=256
export VERBOSE=true

export LOAD_DATA="${WORK_DIR}/pretrained_models/preprocessed_data.pt"
# export LOAD_MODEL="${WORK_DIR}/pretrained_models/mvadapt.pth"
# export SAVE_DATA="${WORK_DIR}/pretrained_models/preprocessed_data.pt"
export SAVE_MODEL="${WORK_DIR}/pretrained_models/mvadapt.pth"

# export LOAD_DATA="None"
export LOAD_MODEL="None"
export SAVE_DATA="None"
# export SAVE_MODEL="None"


python ${WORK_DIR}/team_code/mvadapt_train.py \
--vehicle_indices=${VEHICLE} \
--root_dir=${ROOT_DIR} \
--base_model=${BASE_MODEL} \
--dim0=${DIM0} \
--dim1=${DIM1} \
--dim2=${DIM2} \
--dim3=${DIM3} \
--device=${DEVICE} \
--physics_dim=${PHYSICS_DIM} \
--max_gear_num=${MAX_GEAR_NUM} \
--gear_dim=${GEAR_DIM} \
--epochs=${EPOCHS} \
--lr=${LR} \
--batch_size=${BATCH_SIZE} \
--save_data=${SAVE_DATA} \
--save_model=${SAVE_MODEL} \
--load_data=${LOAD_DATA} \
--load_model=${LOAD_MODEL} \
--verbose=${VERBOSE}
