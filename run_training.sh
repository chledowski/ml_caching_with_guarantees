#!/bin/bash

DATASET=$1
DEVICE=$2
FRACTION=$3
DAGGER=$4
RESULT_DIR=$5

# Train on fraction
if [ "${FRACTION}" != "1" ]; then

    WC=($(wc -l "cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv"))
    NUM_LINES=${WC[0]}
    echo "${NUM_LINES}"
    DECREASED_NUM_LINES=$(echo "scale=4; $NUM_LINES*$FRACTION" | bc)
    echo "${DECREASED_NUM_LINES}"
    DECREASED_NUM_LINES_INT=${DECREASED_NUM_LINES%.*}
    echo "${DECREASED_NUM_LINES_INT}"
    head -n "${DECREASED_NUM_LINES_INT}" "cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv" \
     > "cache_replacement/policy_learning/cache/traces/${DATASET}_train_${FRACTION}.csv"


    if [ "${DAGGER}" == "True" ]; then
        CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
            --experiment_base_dir="${RESULT_DIR}" \
            --experiment_name="${DATASET}__dagger=true__fraction=${FRACTION}" \
            --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
            --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
            --dagger_schedule_bindings="update_freq=5000" \
            --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_train_${FRACTION}.csv" \
            --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
            --total_steps=30001 \
            --save_freq=10000 \
            --full_eval_freq=10000 \
            --small_eval_freq=5000 \
            --small_eval_size=5000 \

    else
        CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
            --experiment_base_dir="${RESULT_DIR}" \
            --experiment_name="${DATASET}__dagger=false__fraction=${FRACTION}" \
            --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
            --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
            --dagger_schedule_bindings="update_freq=100000000000" \
            --dagger_schedule_bindings="initial=0" \
            --dagger_schedule_bindings="final=0" \
            --dagger_schedule_bindings="num_steps=1" \
            --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_train_${FRACTION}.csv" \
            --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
            --total_steps=30001 \
            --save_freq=10000 \
            --full_eval_freq=10000 \
            --small_eval_freq=5000 \
            --small_eval_size=5000 \

    fi
else

    if [ "${DAGGER}" == "True" ]; then
        CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
            --experiment_base_dir="${RESULT_DIR}" \
            --experiment_name="${DATASET}__dagger=true__fraction=${FRACTION}" \
            --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
            --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
            --dagger_schedule_bindings="update_freq=5000" \
            --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv" \
            --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
            --total_steps=30001 \
            --save_freq=10000 \
            --full_eval_freq=10000 \
            --small_eval_freq=5000 \
            --small_eval_size=5000 \

    else
        CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
            --experiment_base_dir="${RESULT_DIR}" \
            --experiment_name="${DATASET}__dagger=false__fraction=${FRACTION}" \
            --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
            --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
            --dagger_schedule_bindings="update_freq=100000000000" \
            --dagger_schedule_bindings="initial=0" \
            --dagger_schedule_bindings="final=0" \
            --dagger_schedule_bindings="num_steps=1" \
            --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv" \
            --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
            --total_steps=30001 \
            --save_freq=10000 \
            --full_eval_freq=10000 \
            --small_eval_freq=5000 \
            --small_eval_size=5000 \

    fi
fi

# Evaluate
CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
            --experiment_base_dir="${RESULT_DIR}" \
            --experiment_name="${DATASET}__dagger=false__fraction=${FRACTION}_eval_test" \
            --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
            --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
            --dagger_schedule_bindings="update_freq=100000000000" \
            --dagger_schedule_bindings="initial=0" \
            --dagger_schedule_bindings="final=0" \
            --dagger_schedule_bindings="num_steps=1" \
            --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
            --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
            --load_checkpoint="${RESULT_DIR}/${DATASET}__dagger=false__fraction=${FRACTION}/checkpoints/20000.ckpt" \
            --total_steps=1

CUDA_VISIBLE_DEVICES=1 python3 -m cache_replacement.policy_learning.cache_model.main \
            --experiment_base_dir="/local/data/chledows/oa" \
            --experiment_name="astar__dagger=true__fraction=0.1_eval_test" \
            --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
            --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
            --dagger_schedule_bindings="update_freq=100000000000" \
            --dagger_schedule_bindings="initial=0" \
            --dagger_schedule_bindings="final=0" \
            --dagger_schedule_bindings="num_steps=1" \
            --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_valid.csv" \
            --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_valid.csv" \
            --load_checkpoint="/local/data/chledows/oa/astar__dagger=true__fraction=0.1/checkpoints/20000.ckpt" \
            --total_steps=1


CUDA_VISIBLE_DEVICES=1 python3 -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir="/local/data/chledows/oa" \
  --experiment_name="astar__dagger=false__fraction=0.001test" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
  --memtrace_file="cache_replacement/policy_learning/cache/traces/astar_valid.csv" \
  --config_bindings="associativity=16" \
  --config_bindings="capacity=2097152" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"/local/data/chledows/oa/astar__dagger=false__fraction=0.001/checkpoints/20000.ckpt\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"/local/data/chledows/oa/astar__dagger=false__fraction=0.001/model_config.json\"" \
  --warmup_period=0