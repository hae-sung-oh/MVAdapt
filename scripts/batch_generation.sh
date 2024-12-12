#!/bin/bash

source set_environment.sh

export PORT="2000"
export TRAFFIC_MANAGER_PORT="2500"

VEHICLES=(1 2 6 7 11 20 24)
SCENES=(4 3 7 8 9 10 1)
TOWNS=(Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD)

for _TOWN in "${TOWNS[@]}"; do
    for _SCENE in "${SCENES[@]}"; do
        for _VEHICLEINDEX in "${VEHICLES[@]}"; do
            export VEHICLEINDEX=$_VEHICLEINDEX
            export SCENE=$_SCENE
            export TOWN=$_TOWN
            echo "Generating data for vehicle index $VEHICLEINDEX, scene $SCENE, town $TOWN"
            ./generate_data.sh
        done
    done
done
