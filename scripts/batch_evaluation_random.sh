#!/bin/bash

source set_environment.sh

# DATE=$(date '+%y%m%d-%H%M')
DATE=250123
export PORT="25000"
export TRAFFIC_MANAGER_PORT="25500"

VEHICLES=(1 2 3 5 6 7 8 9 11 12 15 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 35)

for _VEHICLEINDEX in "${VEHICLES[@]}"; do
  export ADAPT=1
  export ADAPT_PATH="${WORK_DIR}/pretrained_models/mvadapt.pth"
  export RANDOM_PHYSICS=1
  export RANDOM_PHYSICS_PATH="${WORK_DIR}/result/${DATE}/random/physics_${_VEHICLEINDEX}.pickle"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/random/simulation_results_${_VEHICLEINDEX}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/${DATE}/random/result_list_${_VEHICLEINDEX}.pickle"
  export VEHICLEINDEX=$_VEHICLEINDEX
  echo "Starting index $VEHICLEINDEX"
  ./evaluate.sh
done