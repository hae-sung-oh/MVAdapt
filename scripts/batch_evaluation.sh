#!/bin/bash

source set_environment.sh

DATE=$(date '+%y%m%d-%H%M')

for i in {0..36}; do
  echo "Starting index $i"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/simulation_results_${i}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/pkl/result_list_${i}.pickle"
  export VEHICLEINDEX=$i
  ./evaluate.sh
done