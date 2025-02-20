#!/bin/bash

export CARLA_ROOT=/home/ohs-dyros/carla
export WORK_DIR=/home/ohs-dyros/gitRepo/MVAdapt
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.8-linux-x86_64.egg
export TEAM_CODE_ROOT=${WORK_DIR}/team_code
export TEAM_CODE_MVADAPT=${WORK_DIR}/team_code_mvadapt
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT:$TEAM_CODE_ROOT:$TEAM_CODE_MVADAPT:$WORK_DIR