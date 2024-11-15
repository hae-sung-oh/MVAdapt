#!/bin/bash

export RESUME=1
# export VEHICLEINDEX=6

export HOST="127.0.0.1"
# export PORT="20000"
# export TRAFFIC_MANAGER_PORT="20500"
export TRAFFIC_MANAGER_SEED="0"
export DEBUG=0
export TIMEOUT="6000.0"
export RECORD=""
export REPETITIONS=1
export AGENT="${WORK_DIR}/team_code/data_agent.py"
export TRACK="MAP"
export DIRECT=0

export SCENARIO_DIRECTORY="${WORK_DIR}/leaderboard/data/training"
# export SCENE=1
# export TOWN=Town02

export ROUTES="${SCENARIO_DIRECTORY}/routes/s${SCENE}/${TOWN}_Scenario${SCENE}.xml"
export SCENARIOS="${SCENARIO_DIRECTORY}/scenarios/s${SCENE}/${TOWN}_Scenario${SCENE}.json"
export DATAGEN=1
export BENCHMARK=collection
export CHECKPOINT_ENDPOINT=${WORK_DIR}/dataset/Routes_${TOWN}_V${VEHICLEINDEX}/Dataset_generation_${TOWN}_V${VEHICLEINDEX}.json
export SAVE_PATH=${WORK_DIR}/dataset/Routes_${TOWN}_V${VEHICLEINDEX}
export RESULT_LIST="${WORK_DIR}/dataset/Routes_${TOWN}_V${VEHICLEINDEX}/result_list_${VEHICLEINDEX}.pickle"

EXIT_CODE=-1 

while [ $EXIT_CODE -ne 0 ]; do
    echo "Starting the Python script with --resume=${RESUME}"

    python ${WORK_DIR}/leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --host=${HOST}  \
    --port=${PORT}  \
    --trafficManagerPort=${TRAFFIC_MANAGER_PORT}    \
    --trafficManagerSeed=${TRAFFIC_MANAGER_SEED}    \
    --debug=${DEBUG} \
    --timeout=${TIMEOUT} \
    --routes=${ROUTES} \
    --scenarios=${SCENARIOS} \
    --repetitions=${REPETITIONS} \
    --agent=${AGENT} \
    --agent-config=${AGENTCONFIG} \
    --track=${TRACK} \
    --resume=${RESUME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --result-list=${RESULT_LIST} \
    --index=${VEHICLEINDEX} &
    PYTHON_PID=$!

    wait $PYTHON_PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Script exited with code $EXIT_CODE, killing the process and restarting..."
        kill -9 $PYTHON_PID 2>/dev/null 
        RESUME=1
    else
        echo "Script exited with code 0, stopping."
    fi
done
    