#!/bin/bash

source set_environment.sh

export PORT="2000"
export TRAFFIC_MANAGER_PORT="2500"

# VEHICLES=(1 2 6 7 11 20 24)
VEHICLES=(3 5 8 9 11 12 15 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 35)
SCENES=(3 4 7 8 9 1)
TOWNS=(Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD)
# VEHICLES=(1 2 6 7 11 20 24)
# SCENES=(10)
# TOWNS=(Town03 Town04 Town05 Town06 Town07 Town10HD)

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
