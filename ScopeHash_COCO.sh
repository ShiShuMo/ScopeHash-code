#!/bin/bash

################# ScopeHash: Hybrid Training Script for COCO #################

# ========== 核心训练参数 ==========
lr=2e-4
finetune_lr=2e-5
hash_lr=4e-4
gpu_rank=0
epochs=100
batch_size=128
res_name="result/Result_ScopeHash_COCO"

# ========== 架构与微调参数 ==========
# LoRA 和 Unfreezing 设置
lora_rank=16
unfreeze_last_n_layers=0
# 融合模块参数
fusion_num_layers=1
fusion_num_heads=4
# 推理时使用的 Top-K
top_k=5

# ========== 损失权重 ==========
hyper_info_nce=5.0
hyper_csq=2.5
hyper_aware_ce_inter=1.0
hyper_aware_ce_intra=0.001
hyper_quan=1.0
hyper_recon=0.005
mu=20.0
# 必要的分类损失权重
cls_loss_weight=0.2

# ========== 新增参数 ==========
quant_std=0.5
hyper_proxy=2.0
warmup_proportion=0.3
weight_decay=0.01
dropout=0.2
res_mlp_layers=3
activation=gelu
fused_dim=512
residual_weight=0.5
auxiliary_bit_dim=256
concept_num=256
k_bits_list="16,32,64"
tao_global=0.07
tao_local=0.07
drop_path_rate=0.1
drop_rate=0.2

# ========== 训练命令 ==========
echo "🚀 Starting ScopeHash Training on COCO Dataset"
echo "======================================================"
echo "Model: DINOv2 with CAM Guidance"
echo "Learning Rate: $lr"
echo "Finetune LR: $finetune_lr"
echo "Hash LR: $hash_lr"
echo "Batch Size: $batch_size"
echo "Epochs: $epochs"
echo "======================================================"

# ========== AMP 控制 ==========
# 设置为 "bf16", "fp16", 或 "none"
AMP_MODE="bf16"

# 根据AMP_MODE动态添加参数
AMP_ARGS=""
if [ "$AMP_MODE" = "bf16" ]; then
    AMP_ARGS="--use_bf16"
    echo "AMP Mode: bfloat16 enabled"
elif [ "$AMP_MODE" = "fp16" ]; then
    AMP_ARGS="--use_fp16"
    echo "AMP Mode: float16 enabled"
else
    echo "AMP Mode: Disabled (float32)"
fi
echo "======================================================"

python main.py \
    --is-train \
    --dataset coco \
    --result-name "$res_name" \
    --rank "$gpu_rank" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --valid-freq 10 \
    --lr "$lr" \
    --finetune-lr "$finetune_lr" \
    --hash-lr "$hash_lr" \
    --lora-rank "$lora_rank" \
    --unfreeze-last-n-layers "$unfreeze_last_n_layers" \
    --fusion-num-layers "$fusion_num_layers" \
    --fusion-num-heads "$fusion_num_heads" \
    --top-k "$top_k" \
    --hyper-info-nce "$hyper_info_nce" \
    --hyper-csq "$hyper_csq" \
    --hyper-aware-ce-inter "$hyper_aware_ce_inter" \
    --hyper-aware-ce-intra "$hyper_aware_ce_intra" \
    --hyper-quan "$hyper_quan" \
    --hyper-recon "$hyper_recon" \
    --mu "$mu" \
    --cls-loss-weight "$cls_loss_weight" \
    --quant-std "$quant_std" \
    --hyper-proxy "$hyper_proxy" \
    --warmup-proportion "$warmup_proportion" \
    --weight-decay "$weight_decay" \
    --dropout "$dropout" \
    --res-mlp-layers "$res_mlp_layers" \
    --activation "$activation" \
    --fused-dim "$fused_dim" \
    --residual-weight "$residual_weight" \
    --auxiliary-bit-dim "$auxiliary_bit_dim" \
    --concept-num "$concept_num" \
    --k-bits-list "$k_bits_list" \
    --tao-global "$tao_global" \
    --tao-local "$tao_local" \
    --drop-path-rate "$drop_path_rate" \
    --drop-rate "$drop_rate" \
    $AMP_ARGS

echo "✅ ScopeHash COCO Training Completed!"
