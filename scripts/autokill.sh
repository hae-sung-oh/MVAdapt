#!/bin/bash

PORT=${1:-2000}

interval=5  
threshold=20
carla_threshold=10

declare -A count
declare total

total=0  # Initialize total count explicitly

while true; do
    total=0
    pid=$(ps aux | grep leaderboard_evaluator | grep port=${PORT} | grep -v grep | awk '{print $2}' | head -n 1)
    carla=$(ps aux | grep CarlaUE4.sh | grep world-port=${PORT} | grep -v grep | awk '{print $2}' | head -n 1)
    status=$(ps aux | grep leaderboard_evaluator | grep port=${PORT} | grep -v grep | awk '{print $8}' | head -n 1)

    state=""
    if [[ -n "$status" ]]; then
        state=$(echo $status | cut -c 1)
    fi

    printf "\rPID: %s, Status: %s" "$pid" "$status"

    if [[ -n "$pid" && ( "$state" == "S" || "$state" == "D" || "$state" == "X" || "$state" == "Z" ) ]]; then
        if [[ -n "${count[$pid]}" ]]; then
            count[$pid]=$((count[$pid]+1))
        else
            count[$pid]=1
        fi

        if [[ ${count[$pid]} -ge $threshold ]]; then
            echo
            echo -e "Killing $pid : $status"
            kill -9 $pid
            unset count[$pid]
            total=$((total+1))
        fi
    else
        if [[ -n "$pid" ]]; then
            count[$pid]=0
        fi
    fi

    if [[ -n "$pid" && ! $(ps -p $pid > /dev/null 2>&1) ]]; then
        unset count[$pid]
    fi

    if [[ ${total} -ge $carla_threshold && -n "$carla" ]]; then
        echo
        echo "Killing Carla"
        kill -9 $carla
    fi

    sleep $interval
done