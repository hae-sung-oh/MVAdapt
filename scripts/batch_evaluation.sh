#!/bin/bash

source set_environment.sh

DATE=$(date '+%y%m%d-%H%M')
export PORT="2000"
export TRAFFIC_MANAGER_PORT="2500"

VEHICLES=(0 1 2 3 5 6 7 8 9 11 12 13 14 15 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 35)
export SPLIT="trained"

for _VEHICLE_ID in "${VEHICLES[@]}"; do
  export VEHICLE_ID=$_VEHICLE_ID
  export ADAPT=1
  export ADAPT_PATH="${WORK_DIR}/pretrained_models/mvadapt.pth"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLE_ID}/simulation_results_${_VEHICLE_ID}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLE_ID}/result_list_${_VEHICLE_ID}.pickle"
  export DEBUG_PATH="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLE_ID}/debug_${VEHICLE_ID}"
  export STREAM=1
  echo "Starting index $VEHICLE_ID"
  ./evaluate.sh
done

export SPLIT="random"

for _VEHICLE_ID in "${VEHICLES[@]}"; do
  export VEHICLE_ID=$_VEHICLE_ID
  export ADAPT=1
  export ADAPT_PATH="${WORK_DIR}/pretrained_models/mvadapt.pth"
  export RANDOM_PHYSICS=1
  export RANDOM_PHYSICS_PATH="${WORK_DIR}/result/${DATE}/${SPLIT}/physics_${_VEHICLE_ID}.pickle"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLE_ID}/simulation_results_${_VEHICLE_ID}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLE_ID}/result_list_${_VEHICLE_ID}.pickle"
  export DEBUG_PATH="${WORK_DIR}/result/${DATE}/${SPLIT}/V${_VEHICLE_ID}/debug_${VEHICLE_ID}"
  export STREAM=1
  echo "Starting index $VEHICLE_ID"
  ./evaluate.sh
done