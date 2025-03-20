#!/bin/bash

VERSION_LIST=("v5" "v6")
REPEAT=1

for _VERSION in "${VERSION_LIST[@]}"; do
    for _REPEAT in $(seq 1 $REPEAT); do
        export REPEAT=$_REPEAT
        export VERSION=$_VERSION
        echo "Training model for version $VERSION"
        ./train_mvadapt.sh
    done
done
