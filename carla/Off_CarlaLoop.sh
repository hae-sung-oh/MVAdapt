#!/bin/bash

PORT=${1:-2000}

cleanup() {
    echo "Killing Carla server..."
    kill $CARLA_PID
    exit 0
}

trap cleanup SIGINT SIGTERM

while true; do
    ./Dist/CARLA_Shipping_0.9.15-dirty/LinuxNoEditor/CarlaUE4.sh -RenderOffScreen --world-port=$PORT \
#    ./Dist/CARLA_Shipping_0.9.12-dirty/LinuxNoEditor/CarlaUE4.sh  --world-port=$PORT \
    2> >(grep -v -e "ERROR: Invalid session: no stream available with id" \
    -e "error retrieving stream id" \
    -e "ERROR" \
    -e "Connection reset by peer" \
    -e "End of file") & CARLA_PID=$!

    wait $CARLA_PID

    echo -e "\nCarla Server has stopped. Restarting...\n"
    sleep 2
done
