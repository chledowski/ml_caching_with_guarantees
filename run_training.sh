#!/bin/bash

DATASET=$1
DEVICE=$2
FRACTION=$3
DAGGER=$4
RESULT_DIR=$5
DELETE_PARSED_OUTS=$6  # we parse the outputs from the training. They take much space. This enables their deletion.

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

    TRAIN_TRACE="cache_replacement/policy_learning/cache/traces/${DATASET}_train_${FRACTION}.csv"

else
    TRAIN_TRACE="cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv"
fi

if [ "${DAGGER}" == "True" ]; then
    EXP_NAME="${DATASET}__dagger=true__fraction=${FRACTION}"

    CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
        --experiment_base_dir="${RESULT_DIR}" \
        --experiment_name="${EXP_NAME}" \
        --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
        --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
        --dagger_schedule_bindings="update_freq=5000" \
        --train_memtrace="${TRAIN_TRACE}" \
        --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
        --total_steps=20001 \
        --save_freq=10000 \
        --full_eval_freq=10000 \
        --small_eval_freq=5000 \
        --small_eval_size=5000

else
    EXP_NAME="${DATASET}__dagger=false__fraction=${FRACTION}"
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
        --experiment_base_dir="${RESULT_DIR}" \
        --experiment_name="${EXP_NAME}" \
        --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
        --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
        --dagger_schedule_bindings="update_freq=100000000000" \
        --dagger_schedule_bindings="initial=0" \
        --dagger_schedule_bindings="final=0" \
        --dagger_schedule_bindings="num_steps=1" \
        --train_memtrace="${TRAIN_TRACE}" \
        --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
        --total_steps=20001 \
        --save_freq=10000 \
        --full_eval_freq=10000 \
        --small_eval_freq=5000 \
        --small_eval_size=5000

fi


# ADD CHOOSING BEST CKPT!

# TODO change 1 valid to test below
# Evaluate
CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
            --experiment_base_dir="${RESULT_DIR}/${EXP_NAME}" \
            --experiment_name="test" \
            --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
            --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
            --dagger_schedule_bindings="update_freq=100000000000" \
            --dagger_schedule_bindings="initial=0" \
            --dagger_schedule_bindings="final=0" \
            --dagger_schedule_bindings="num_steps=1" \
            --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
            --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
            --evaluate=True \
            --total_steps=1

# TODO: parse

#CUDA_VISIBLE_DEVICES=6 python3 -m cache_replacement.policy_learning.cache_model.main \
#            --experiment_base_dir="/local/data/chledows/oa/testing_1/astar__dagger=false__fraction=1/" \
#            --experiment_name="test" \
#            --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
#            --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
#            --dagger_schedule_bindings="update_freq=100000000000" \
#            --dagger_schedule_bindings="initial=0" \
#            --dagger_schedule_bindings="final=0" \
#            --dagger_schedule_bindings="num_steps=1" \
#            --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_valid.csv" \
#            --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_valid.csv" \
#            --load_checkpoint="/local/data/chledows/oa/testing_1/astar__dagger=false__fraction=1/checkpoints/10000.ckpt" \
#            --total_steps=1


#CUDA_VISIBLE_DEVICES=1 python3 -m cache_replacement.policy_learning.cache.main \
#  --experiment_base_dir="/local/data/chledows/oa" \
#  --experiment_name="astar__dagger=false__fraction=0.001test" \
#  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
#  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
#  --memtrace_file="cache_replacement/policy_learning/cache/traces/astar_valid.csv" \
#  --config_bindings="associativity=16" \
#  --config_bindings="capacity=2097152" \
#  --config_bindings="eviction_policy.scorer.checkpoint=\"/local/data/chledows/oa/old/astar__dagger=false__fraction=0.1/checkpoints/20000.ckpt\"" \
#  --config_bindings="eviction_policy.scorer.config_path=\"/local/data/chledows/oa/astar__dagger=false__fraction=0.1/model_config.json\"" \
#  --warmup_period=0
#
#
#CUDA_VISIBLE_DEVICES=7 python3 -m cache_replacement.policy_learning.cache_model.main \
#    --experiment_base_dir="/local/data/chledows/oa" \
#    --experiment_name="astar__test_training_with_logging" \
#    --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
#    --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
#    --dagger_schedule_bindings="update_freq=5000" \
#    --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_train.csv" \
#    --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_valid.csv" \
#    --total_steps=30001 \
#    --save_freq=10000 \
#    --full_eval_freq=10000 \
#    --small_eval_freq=5000 \
#    --small_eval_size=5000
#
#
#CUDA_VISIBLE_DEVICES=7 python3 -m cache_replacement.policy_learning.cache_model.main \
#  --experiment_base_dir=/tmp \
#  --experiment_name=sample_model_llc \
#  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
#  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
#  --model_bindings="address_embedder.max_vocab_size=5000" \
#  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
#  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
#    --total_steps=3001 \
#    --save_freq=1000 \
#    --full_eval_freq=1000 \
#    --small_eval_freq=500 \
#    --small_eval_size=500