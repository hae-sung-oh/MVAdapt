#!/bin/bash

source set_environment.sh

# DATE=$(date '+%y%m%d-%H%M')
DATE="250317"
export PORT="2000"
export TRAFFIC_MANAGER_PORT="2500"

# VEHICLES=(1 2 3 5 6 7 8 9 11 12 13 14 15 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 35)
VEHICLES=(5 8 14 23 24 31 32 33 35 0) #7
export SPLIT="trained"

for _VEHICLEINDEX in "${VEHICLES[@]}"; do
  export VEHICLEINDEX=$_VEHICLEINDEX
  export ADAPT=1
  export ADAPT_PATH="${WORK_DIR}/pretrained_models/mvadapt_v4_1.pth"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/simulation_results_${_VEHICLEINDEX}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/result_list_${_VEHICLEINDEX}.pickle"
  export DEBUG_PATH="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/debug_${VEHICLEINDEX}"
  export STREAM=1
  echo "Starting index $VEHICLEINDEX"
  ./evaluate.sh
done

VEHICLES=(0 1 2 3 5 6 7 8 9 11 12 13 14 15 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 35)
export SPLIT="untrained"

for _VEHICLEINDEX in "${VEHICLES[@]}"; do
  export VEHICLEINDEX=$_VEHICLEINDEX
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/simulation_results_${_VEHICLEINDEX}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/result_list_${_VEHICLEINDEX}.pickle"
  export DEBUG_PATH="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/debug_${VEHICLEINDEX}"
  export STREAM=1
  echo "Starting index $VEHICLEINDEX"
  ./evaluate.sh
done

VEHICLES=(0 1 2 3 5 6 7 8 9 11 12 13 14 15 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 35)
export SPLIT="random"

for _VEHICLEINDEX in "${VEHICLES[@]}"; do
  export VEHICLEINDEX=$_VEHICLEINDEX
  export ADAPT=1
  export ADAPT_PATH="${WORK_DIR}/pretrained_models/mvadapt_v4_1.pth"
  export RANDOM_PHYSICS=1
  export RANDOM_PHYSICS_PATH="${WORK_DIR}/result/${DATE}/${SPLIT}/physics_${_VEHICLEINDEX}.pickle"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/simulation_results_${_VEHICLEINDEX}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/result_list_${_VEHICLEINDEX}.pickle"
  export DEBUG_PATH="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLEINDEX}/debug_${VEHICLEINDEX}"
  export STREAM=1
  echo "Starting index $VEHICLEINDEX"
  ./evaluate.sh
done