#!/bin/bash

source set_environment.sh

for i in {1..36}; do
  echo "Starting index $i"
  export CHECKPOINT="${WORK_DIR}/result/simulation_results_${i}.json"
  export RESULT_LIST="${WORK_DIR}/result/result_list_${i}.pickle"
  export VEHICLEINDEX=$i
  ./evaluate.sh
done