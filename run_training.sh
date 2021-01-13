#!/bin/bash

DATASET=$1
DEVICE=$2
FRACTION=$3
DAGGER=$4
RESULT_DIR=$5

if [ "${FRACTION}" != "100" ]; then
    WC=($(wc -l "cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv"))
#    echo $WC
    NUM_LINES=${WC[0]}
    echo $NUM_LINES

#    NUM_LINES=10
    ans=$((NUM_LINES * FRACTION))
    echo $ans
#    A=$(($FRACTION * $NUM_LINES))
#    B=$(( $A / 100 ))
#    echo $B
#    echo $NUM_LINES
#    echo $FRACTION

fi

#if [ "${DAGGER}" == "True" ]; then
#
#    CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
#        --experiment_base_dir="${RESULT_DIR}" \
#        --experiment_name="${DATASET}__dagger=true__fraction=${FRACTION}" \
#        --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
#        --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
#        --dagger_schedule_bindings="update_freq=5000" \
#        --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv" \
#        --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
#        --total_steps=1500000
#
#else
#
#    CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
#        --experiment_base_dir="${RESULT_DIR}" \
#        --experiment_name="${DATASET}__dagger=false__fraction=${FRACTION}" \
#        --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
#        --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
#        --dagger_schedule_bindings="update_freq=100000000000" \
#        --dagger_schedule_bindings="initial=0" \
#        --dagger_schedule_bindings="final=0" \
#        --dagger_schedule_bindings="num_steps=1" \
#        --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv" \
#        --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
#        --total_steps=1500000
#
#fi