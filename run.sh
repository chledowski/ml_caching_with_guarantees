#!/bin/bash

DATASET=$1
DEVICE=$2
FRACTION=$3
DAGGER=$4
RESULT_DIR=$5

# Train on fraction
if [ "${FRACTION}" != "1" ]; then

#    WC=($(wc -l "cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv"))
#    NUM_LINES=${WC[0]}
#    echo "${NUM_LINES}"
#    DECREASED_NUM_LINES=$(echo "scale=4; $NUM_LINES*$FRACTION" | bc)
#    echo "${DECREASED_NUM_LINES}"
#    DECREASED_NUM_LINES_INT=${DECREASED_NUM_LINES%.*}
#    echo "${DECREASED_NUM_LINES_INT}"
#    head -n "${DECREASED_NUM_LINES_INT}" "cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv" \
#     > "cache_replacement/policy_learning/cache/traces/${DATASET}_train_${FRACTION}.csv"

    TRAIN_TRACE="cache_replacement/policy_learning/cache/traces/${DATASET}_train_${FRACTION}.csv"

else
    TRAIN_TRACE="cache_replacement/policy_learning/cache/traces/${DATASET}_train.csv"
fi

if [ "${DAGGER}" == "True" ]; then
    EXP_NAME="${DATASET}__dagger=true__fraction=${FRACTION}"

#    CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
#        --experiment_base_dir="${RESULT_DIR}" \
#        --experiment_name="${EXP_NAME}" \
#        --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
#        --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
#        --model_bindings="address_embedder.max_vocab_size=5000" \
#        --dagger_schedule_bindings="update_freq=5000" \
#        --train_memtrace="${TRAIN_TRACE}" \
#        --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
#        --total_steps=20001 \
#        --save_freq=5000 \
#        --full_eval_freq=5000

else
    EXP_NAME="${DATASET}__dagger=false__fraction=${FRACTION}"
#    CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
#        --experiment_base_dir="${RESULT_DIR}" \
#        --experiment_name="${EXP_NAME}" \
#        --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
#        --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
#        --model_bindings="address_embedder.max_vocab_size=5000" \
#        --dagger_schedule_bindings="update_freq=100000000000" \
#        --dagger_schedule_bindings="initial=0" \
#        --dagger_schedule_bindings="final=0" \
#        --dagger_schedule_bindings="num_steps=1" \
#        --train_memtrace="${TRAIN_TRACE}" \
#        --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_valid.csv" \
#        --total_steps=20001 \
#        --save_freq=10000 \
#        --full_eval_freq=10000

fi

# Evaluate
CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m cache_replacement.policy_learning.cache_model.main \
    --experiment_base_dir="${RESULT_DIR}/${EXP_NAME}" \
    --experiment_name="test" \
    --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
    --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
    --model_bindings="address_embedder.max_vocab_size=5000" \
    --dagger_schedule_bindings="update_freq=100000000000" \
    --dagger_schedule_bindings="initial=0" \
    --dagger_schedule_bindings="final=0" \
    --dagger_schedule_bindings="num_steps=1" \
    --train_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_test.csv" \
    --valid_memtrace="cache_replacement/policy_learning/cache/traces/${DATASET}_test.csv" \
    --evaluate=True \
    --total_steps=1

mkdir -p "${RESULT_DIR}/parsed/"
mkdir -p "${RESULT_DIR}/parsed/${EXP_NAME}"
# Parse the outs
python3 parse_outs.py --exp-folder="${RESULT_DIR}/${EXP_NAME}/test" --steps=0 --out-folder="${RESULT_DIR}/parsed/${EXP_NAME}"

# Move logs to the parsed folder
cp -r "${RESULT_DIR}/${EXP_NAME}/test/logs.txt" "${RESULT_DIR}/parsed/${EXP_NAME}/logs.txt"
