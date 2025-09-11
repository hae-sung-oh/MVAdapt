#!/bin/bash

PORT=${1:-2000}
CARLA_DIST_DIR='/path/to/your/carla/Dist/CARLA_Shipping_0.9.12-dirty/LinuxNoEditor'

cleanup() {
    echo "Killing Carla server..."
    kill $CARLA_PID
    exit 0
}

trap cleanup SIGINT SIGTERM

while true; do
    .${CARLA_DIST_DIR}/CarlaUE4.sh  --world-port=$PORT \
    & CARLA_PID=$!

    wait $CARLA_PID

    echo -e "\nCarla Server has stopped. Restarting...\n"
    sleep 2
done
