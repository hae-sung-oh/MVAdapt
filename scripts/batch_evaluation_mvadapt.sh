#!/bin/bash

source set_environment.sh

DATE=$(date '+%y%m%d-%H%M')

cd ..
python reset_result.py
cd scripts/

for i in {7..36}; do
  echo "Starting index $i"
  export ADAPT=0
  export DIM0=32
  export DIM1=32
  export ADAPT_PATH="${WORK_DIR}/pretrained_models/mvadapt.pth"
  export CHECKPOINT="${WORK_DIR}/result/${DATE}/simulation_results_${i}_${DATE}.json"
  export RESULT_LIST="${WORK_DIR}/result/pkl/result_list_${i}.pickle"
  export VEHICLEINDEX=$i
  ./evaluate.sh
done