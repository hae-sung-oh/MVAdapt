export CARLA_ROOT=/home/ohs-dyros/carla
export WORK_DIR=/home/ohs-dyros/gitRepo/carla_garage
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/team_code
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# export ROUTE="${WORK_DIR}/leaderboard/data/longest6.xml"
export ROUTE="${WORK_DIR}/leaderboard/data/lav.xml"

export AGENTCONFIG="${WORK_DIR}/pretrained_models/lav/aim_02_05_withheld_0"
# export AGENTCONFIG="${WORK_DIR}/pretrained_models/leaderboard/tfpp_wp_all_0"
# export AGENTCONFIG="${WORK_DIR}/pretrained_models/longest6/plant_all_1"

export HOST="127.0.0.1"
export PORT="2000"
export TRAFFIC_MANAGER_PORT="2500"
export TRAFFIC_MANAGER_SEED="0"
export DEBUG=0
export TIMEOUT="6000.0"
export RECORD=""
export SCENARIOS="${WORK_DIR}/leaderboard/data/scenarios/eval_scenarios.json"
export REPETITIONS=1
export AGENT="${WORK_DIR}/team_code/sensor_agent.py"
export TRACK="SENSORS"
export DIRECT=0
export RESUME=1
# export CHECKPOINT="${WORK_DIR}/result/simulation_results.json"
# export RESULT_LIST="${WORK_DIR}/result_list.pickle"

EXIT_CODE=-1 

# export VEHICLEMODEL="vehicle.lincoln.mkz_2017"

while [ $EXIT_CODE -ne 0 ]; do
    echo "Starting the Python script with --resume=${RESUME}"

    python ${WORK_DIR}/leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --host=${HOST}  \
    --port=${PORT}  \
    --trafficManagerPort=${TRAFFIC_MANAGER_PORT}    \
    --trafficManagerSeed=${TRAFFIC_MANAGER_SEED}    \
    --debug=${DEBUG} \
    --timeout=${TIMEOUT} \
    --routes=${ROUTE} \
    --scenarios=${SCENARIOS} \
    --repetitions=${REPETITIONS} \
    --agent=${AGENT} \
    --agent-config=${AGENTCONFIG} \
    --track=${TRACK} \
    --resume=${RESUME} \
    --checkpoint=${CHECKPOINT} \
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
    