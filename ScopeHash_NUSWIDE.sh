#!/bin/bash

################# ScopeHash: Stabilized Training Script for NUS-WIDE #################

# ========== 优化后的核心训练参数 ==========
lr=1.5e-4                    # 降低基础学习率
finetune_lr=2e-5             # 降低预训练部分学习率
hash_lr=3e-4                 # 降低哈希学习率
gpu_rank=0
epochs=80                    # 减少epoch数，避免过拟合
batch_size=128                # 增加batch_size提高稳定性
res_name="result/Result_ScopeHash_NUSWIDE"

# ========== 架构与微调参数 ==========
lora_rank=16
unfreeze_last_n_layers=0
fusion_num_layers=1
fusion_num_heads=4
top_k=4

# ========== 优化后的损失权重 ==========
hyper_info_nce=0.7           # 降低对比学习权重
hyper_csq=3.5
hyper_aware_ce_inter=1.0
hyper_aware_ce_intra=0.001
hyper_quan=1.0               # 降低量化损失权重
hyper_recon=0.002
mu=18.0                      # 降低mu值
cls_loss_weight=0.25          # 增加分类损失权重

# ========== 稳定性优化参数 ==========
quant_std=0.3                # 降低量化标准差
hyper_proxy=1.5              # 降低代理损失权重
warmup_proportion=0.3       # 大幅减少warmup
weight_decay=0.08            #大幅降低权重衰减
dropout=0.18                 # 降低dropout
res_mlp_layers=2
activation=gelu
fused_dim=512
residual_weight=0.5
auxiliary_bit_dim=256
concept_num=256
k_bits_list="16,32,64"
tao_global=0.05              # 降低温度参数
tao_local=0.05
drop_path_rate=0.1
drop_rate=0.2

# ========== 训练命令 ==========
echo "🚀 Starting Stabilized ScopeHash Training on NUS-WIDE Dataset"
echo "======================================================"
echo "Stabilization Focus: Reduced regularization + shorter warmup"
echo "Learning Rate: $lr (reduced)"
echo "Finetune LR: $finetune_lr (reduced)"
echo "Hash LR: $hash_lr (reduced)"
echo "Batch Size: $batch_size (increased)"
echo "Warmup: $warmup_proportion (much shorter)"
echo "Weight Decay: $weight_decay (much lower)"
echo "Dropout: $dropout (much lower)"
echo "Epochs: $epochs (reduced)"
echo "======================================================"

# ========== AMP 控制 ==========
AMP_MODE="bf16"

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
    --dataset nuswide \
    --result-name "$res_name" \
    --rank "$gpu_rank" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --valid-freq 8 \
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

echo "✅ Stabilized ScopeHash NUS-WIDE Training Completed!"
