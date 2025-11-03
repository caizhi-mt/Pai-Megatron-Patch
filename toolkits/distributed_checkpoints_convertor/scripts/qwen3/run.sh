#!/bin/bash
set -e
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CONVERTOR_DIR}))
#export PYTHONPATH=${MEGATRON_PATCH_PATH}:/mnt/seed-program-nas/001688/caizhi/qwen3_train/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624/:${CONVERTOR_DIR}/impl:/mnt/seed-program-nas/001688/caizhi/qwen3_train/megatron-lm-musa-patch:$PYTHONPATH
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${CONVERTOR_DIR}/impl:/mnt/seed-program-nas/001688/caizhi/qwen3_train/Megatron-LM:/mnt/seed-program-nas/001688/caizhi/qwen3_train/megatron-lm-musa-patch:$PYTHONPATH
#export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6


#NUM_NODES=${WORLD_SIZE:-1}
NUM_NODES=1
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

MODEL_SIZE="A3B"
#LOAD_DIR="/mnt/seed17/001688/caizhi/Qwen3-30B-A3B-Base/"
#SAVE_DIR="/mnt/seed17/001688/caizhi/Qwen3-30B-A3B-Base-tp2-pp4-ep8-mcore"
#MG2HF=false
#HF_DIR=



#LOAD_DIR="/mnt/seed17/001688/caizhi/output_train_ckpt_CPT/checkpoint/pr-bf16-tp-2-pp-4-cp-1-ac-full-do-true-sp-true-ti-2200-wi--2025-10-25_14:10:51/"
#SAVE_DIR="/mnt/seed17/001688/caizhi/mnt/seed17/001688/caizhi/mcore_hf_ckpt_tp2_pp4_ep8_iter2200_CPT_202510261203"
#LOAD_DIR="/mnt/seed17/001688/caizhi/output_train_ckpt_SFT/checkpoint/pr-bf16-tp-2-pp-4-cp-1-ac-full-do-true-sp-true-ti-3510-wi--2025-10-31_19:25:28/"
LOAD_DIR="/mnt/seed17/001688/caizhi/output_train_ckpt_SFT/checkpoint/pr-bf16-tp-2-pp-4-cp-1-ac-full-do-true-sp-true-ti-3510-wi--2025-11-02_20:27:23/"
SAVE_DIR="/mnt/seed17/001688/caizhi/SFT_pr_bf16tp2pp4cp1_ac_full_do_true_sp_true_3510step_mcore_huggingface"
MG2HF=true
HF_CKPT_PATH="/mnt/seed-program-nas/001688/caizhi/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/Qwen3-30B-A3B"
HF_DIR=${HF_CKPT_PATH}



#MODEL_SIZE="1.7B"
#LOAD_DIR="/mnt/seed-program-nas/001688/caizhi/Qwen3-1.7B"
#SAVE_DIR="/mnt/seed-program-nas/001688/caizhi/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/scripts/qwen3/qwen3_a3b_mcore_ckpt"
#SAVE_DIR="/mnt/seed-program-nas/001688/caizhi/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/scripts/qwen3/qwen3_1_7_b_ckpt"
USE_CUDA=true
PR=bf16

OTHER_ARGS=()
if [ ${MG2HF} = true ]; then
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${HF_DIR}
        --hf-dir ${HF_DIR}
        --mcore2hf
        --hf-ckpt-path ${HF_CKPT_PATH}
    )
    mkdir -p ${SAVE_DIR}
    find -L ${HF_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${HF_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVE_DIR}
else
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${LOAD_DIR}
    )
    mkdir -p ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVE_DIR}
fi

if [ ${USE_CUDA} = true ]; then
    OTHER_ARGS+=(
        --use-gpu
    )
fi

if [ ${PR} = fp16 ]; then
    OTHER_ARGS+=(
        --fp16
    )
elif [ ${PR} = bf16 ]; then
    OTHER_ARGS+=(
        --bf16
    )
fi

if [ -z ${NUM_NODES} ]; then
    echo "Please Provide WORLD_SIZE"
    exit
fi

if [ -z ${NODE_RANK} ]; then
    echo "Please Provide RANK"
    exit
fi

if [ -z ${MASTER_ADDR} ]; then
    echo "Please Provide MASTER_ADDR"
    exit
fi

if [ -z ${MASTER_PORT} ]; then
    echo "Please Provide MASTER_PORT"
    exit
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --seq-length 1
    --max-position-embeddings 40960
    --attention-backend auto # Can use (flash/fused/unfused/local)
    --position-embedding-type rope
    --kv-channels 128
    --qk-layernorm
    --group-query-attention
)

if [ $MODEL_SIZE = 0.6B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 28
        --hidden-size 1024
        --ffn-hidden-size 3072
        --num-attention-heads 16
        --num-query-groups 8
    )
    if [ -z  "$MODEL_PARALLEL_ARGS" ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 4
        )
    fi
elif [ $MODEL_SIZE = 1.7B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 28
        --hidden-size 2048
        --ffn-hidden-size 6144
        --num-attention-heads 16
        --num-query-groups 8
    )
    if [ -z  "$MODEL_PARALLEL_ARGS" ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 1
        )
    fi
elif [ $MODEL_SIZE = 4B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 36
        --hidden-size 2560
        --ffn-hidden-size 9728
        --num-attention-heads 32
        --num-query-groups 8
    )
    if [ -z  "$MODEL_PARALLEL_ARGS" ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 4
        )
    fi
elif [ $MODEL_SIZE = 8B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 36
        --hidden-size 4096
        --ffn-hidden-size 12288
        --num-attention-heads 32
        --untie-embeddings-and-output-weights
        --num-query-groups 8
    )
    if [ -z  "$MODEL_PARALLEL_ARGS" ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 1
        )
    fi
elif [ $MODEL_SIZE = 14B ]; then 
    GPT_MODEL_ARGS+=(
        --num-layers 40
        --hidden-size 5120
        --ffn-hidden-size 17408
        --num-attention-heads 40
        --untie-embeddings-and-output-weights
        --num-query-groups 8
    )
    if [ -z  "$MODEL_PARALLEL_ARGS" ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 8
        )
    fi
elif [ $MODEL_SIZE = 32B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 64
        --hidden-size 5120
        --ffn-hidden-size 25600
        --num-attention-heads 64
        --untie-embeddings-and-output-weights
        --num-query-groups 8
    )
    if [ -z  "$MODEL_PARALLEL_ARGS" ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 8
        )
    fi
elif [ $MODEL_SIZE = A3B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 48
        --hidden-size 2048
        --ffn-hidden-size 6144
        --moe-ffn-hidden-size 768
        --num-attention-heads 32
        --untie-embeddings-and-output-weights
        --moe-grouped-gemm
        --moe-router-score-function softmax
        --moe-token-dispatcher-type alltoall
        --moe-router-topk 8
        --moe-layer-freq "'([1]*48)'"
        --num-experts 128
        --num-query-groups 4
    )
    if [ -z  "$MODEL_PARALLEL_ARGS" ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 1
            --expert-model-parallel-size 1
        )
    fi
elif [ $MODEL_SIZE = A22B ]; then
    echo "Please use 16xH20 for conversion"
    exit
fi

TARGET_ARGS=(
            --target-tensor-model-parallel-size 2
            --target-pipeline-model-parallel-size 4
            --target-expert-model-parallel-size 8
            --target-expert-tensor-parallel-size 1
            #--target-decoder-first-pipeline-num-layers 1
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1024
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 10
)

CONVERT_ARGS=(
    --model-type GPT 
    --load-dir ${LOAD_DIR}
    --save-dir ${SAVE_DIR}
    
    --padded-vocab-size 151936
    --no-load-optim
    --no-load-rng
    --logging-level 5
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} impl/convert_qwen3.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${CONVERT_ARGS[@]} \
    ${TARGET_ARGS[@]} \
    ${OTHER_ARGS[@]}"

echo $cmd
eval $cmd
