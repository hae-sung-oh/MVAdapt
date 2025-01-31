#!/bin/bash

PORT=${1:-2000}

interval=5  
threshold=20
carla_threshold=10

declare -A count
declare total

while true; do
    total=0
    pid=$(ps aux | grep leaderboard_evaluator | grep port=${PORT} | grep -v grep | awk '{print $2}' | head -n 1)
    carla=$(ps aux | grep CarlaUE4.sh | grep world-port=${PORT} | grep -v grep | awk '{print $2}' | head -n 1)
    status=$(ps aux | grep leaderboard_evaluator | grep port=${PORT} | grep -v grep | awk '{print $8}' | head -n 1)

    state=$(echo $status | cut -c 1)

    printf "\rPID: %s, Status: %s" "$pid" "$status"

    if [[ "$state" == "S" || "$state" == "D" || "$state" == "X" || "$state" == "Z" ]]; then
        count[$pid]=$((count[$pid]+1))

        if [[ ${count[$pid]} -ge $threshold ]]; then
            echo
            echo -e "Killing $pid : $status"
            kill -9 $pid
            unset count[$pid]
            total=$((total+1))
        fi
    else
        count[$pid]=0
    fi

    if ! ps -p $pid > /dev/null; then
        unset count[$pid]
    fi

    if [[ ${total} -ge $carla_threshold ]]; then
        echo
        echo "Killing Carla"
        kill -9 $carla
    fi

    sleep $interval
done
