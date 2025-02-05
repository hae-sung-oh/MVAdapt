#!/bin/bash

source set_environment.sh

# DATE=$(date '+%y%m%d-%H%M')
DATE=250205
export PORT="20000"
export TRAFFIC_MANAGER_PORT="20500"

VEHICLES=(21 22 23 24 25 26 27 28 29 30 31 32 33 35)

for _VEHICLEINDEX in "${VEHICLES[@]}"; do
  export ADAPT=1
  export ADAPT_PATH="${WORK_DIR}/pretrained_models/mvadapt_v3_1.pth"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/trained/simulation_results_${_VEHICLEINDEX}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/${DATE}/trained/result_list_${_VEHICLEINDEX}.pickle"
  export VEHICLEINDEX=$_VEHICLEINDEX
  echo "Starting index $VEHICLEINDEX"
  ./evaluate.sh
done

for _VEHICLEINDEX in "${VEHICLES[@]}"; do
  export ADAPT=1
  export ADAPT_PATH="${WORK_DIR}/pretrained_models/mvadapt_v3_1.pth"
  export RANDOM_PHYSICS=1
  export RANDOM_PHYSICS_PATH="${WORK_DIR}/result/${DATE}/random/physics_${_VEHICLEINDEX}.pickle"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/random/simulation_results_${_VEHICLEINDEX}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/${DATE}/random/result_list_${_VEHICLEINDEX}.pickle"
  export VEHICLEINDEX=$_VEHICLEINDEX
  echo "Starting index $VEHICLEINDEX"
  ./evaluate.sh
done