#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
source "$SCRIPT_DIR/../scripts/set_environment.sh"

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.

torchrun    \
    --nnodes=1    \
    --nproc_per_node=1    \
    --max_restarts=1    \
    --rdzv_id=42353467    \
    --rdzv_backend=c10d    \
    train.py    \
    --id train_id_000    \
    --batch_size 8    \
    --setting all    \
    --root_dir ${WORK_DIR}/dataset    \
    --logdir ${WORK_DIR}/logs    \
    --use_controller_input_prediction 1    \
    --use_wp_gru 0    \
    --use_discrete_command 1    \
    --use_tp 1    \
    --continue_epoch 1    \
    --cpu_cores 20    \
    --num_repetitions 3    \

