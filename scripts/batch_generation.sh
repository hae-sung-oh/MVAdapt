#!/bin/bash

source set_environment.sh

(  
    # vehicle.audi.a2
    export PORT="2000"
    export TRAFFIC_MANAGER_PORT="2500"
    export VEHICLEINDEX=1
    export SCENE=1
    export TOWN=Town02
    ./generate_data.sh
)   &
(   # vehicle.carlamotors.carlacola
    export PORT="2000"
    export TRAFFIC_MANAGER_PORT="2500"
    export VEHICLEINDEX=6
    export SCENE=1
    export TOWN=Town02
    ./generate_data.sh
)   &
(  
    # vehicle.carlamotors.firetruck
    export PORT="2000"
    export TRAFFIC_MANAGER_PORT="2500"
    export VEHICLEINDEX=7
    export SCENE=1
    export TOWN=Town02
    ./generate_data.sh
)   
(   # vehicle.citroen.c3
    export PORT="2000"
    export TRAFFIC_MANAGER_PORT="2500"
    export VEHICLEINDEX=9
    export SCENE=1
    export TOWN=Town02
    ./generate_data.sh
)
