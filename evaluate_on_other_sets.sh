#!/bin/bash

DATASET=$1
DEVICE=$2
FRACTION=$3
RESULT_DIR=$5
STEPS=$6

EXP_NAME="${DATASET}__dagger=false__fraction=${FRACTION}_${STEPS}"

# Evaluate
CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
    --experiment_base_dir="${RESULT_DIR}/${EXP_NAME}" \
    --experiment_name="test_other_set" \
    --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
    --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
    --model_bindings="address_embedder.max_vocab_size=5000" \
    --train_memtrace="/local/data/chledows/oa/datasets/${DATASET}/test.csv" \
    --valid_memtrace="/local/data/chledows/oa/datasets/${DATASET}/test.csv" \
    --evaluate=True \
    --total_steps=1

mkdir -p "${RESULT_DIR}/parsed_other_set/"
mkdir -p "${RESULT_DIR}/parsed_other_set/${EXP_NAME}"

# Parse the outs
python3 parse_outs.py --exp-folder="${RESULT_DIR}/${EXP_NAME}/test_other_set" --steps=0 --out-folder="${RESULT_DIR}/parsed_other_set/${EXP_NAME}"

# Move logs to the parsed folder
cp -r "${RESULT_DIR}/${EXP_NAME}/test_other_set/logs.txt" "${RESULT_DIR}/parsed_other_set/${EXP_NAME}/logs.txt"
cp -r "${RESULT_DIR}/${EXP_NAME}/test_other_set/tensorboard" "${RESULT_DIR}/parsed_other_set/${EXP_NAME}/tensorboard_test"
