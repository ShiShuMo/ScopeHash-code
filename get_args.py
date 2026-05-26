import argparse
import datetime
import os

def get_args():
    parser = argparse.ArgumentParser(description="ScopeHash")

    # ========== 基础哈希参数 ==========
    parser.add_argument("--k-bits-list", type=str, default="16,32,64",
                        help="Multi-bit hash code lengths (comma-separated)")
    parser.add_argument("--auxiliary-bit-dim", type=int, default=256,
                        help="Auxiliary hash code dimension for reconstruction loss")

    # ========== 网络架构参数 ==========
    parser.add_argument("--activation", type=str, default="gelu",
                        choices=["relu", "gelu"],
                        help="Activation function for MLPs")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for MLPs")
    parser.add_argument("--res-mlp-layers", type=int, default=2,
                        help="Number of layers in ResidualMLPs")
    parser.add_argument("--fused-dim", type=int, default=512,
                        help="Feature dimension after vision backbone and before fusion")
    parser.add_argument("--residual-weight", type=float, default=0.5,
                        help="Residual weight for feature fusion")

    # ========== 训练参数 ==========
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Base learning rate for newly added modules")
    parser.add_argument("--finetune-lr", type=float, default=2e-4,
                        help="Base learning rate for newly added modules")
    parser.add_argument("--hash-lr", type=float, default=2e-4,
                        help="Base learning rate for newly added modules")
    parser.add_argument("--valid-freq", type=int, default=3,
                        help="Validation frequency (epochs)")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--weight-decay", type=float, default=0.2,
                        help="Weight decay for optimizer")
    parser.add_argument("--concept-num", type=int, default=260)
    parser.add_argument("--warmup-proportion", type=float, default=0.1,
                        help="Warmup proportion for learning rate scheduler")
    parser.add_argument("--use_bf16", 
                        action="store_true", 
                        help="Enable bfloat16 Automatic Mixed Precision (AMP). If not present, AMP is disabled.")
    parser.add_argument("--use_fp16", 
                        action="store_true", 
                        help="Enable float16 Automatic Mixed Precision (AMP). If not present, AMP is disabled.")
    parser.add_argument("--drop-path-rate", type=float, default=0.2,
                        help="Dropout rate for backbone")
    parser.add_argument("--drop-rate", type=float, default=0.2,
                        help="Dropout rate for backbone")

    # ========== 数据集参数 ==========
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["coco", "imagenet", "nuswide"],
                        help="Dataset name")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Image resolution")
    parser.add_argument("--num-workers", type=int, default=12,
                        help="Number of data loading workers")

    # ========== 骨干网络与微调参数 ==========
    parser.add_argument("--siglip-model", type=str, default="",
                        help="SigLIP model variant")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=2,
                        help="Number of last transformer layers to unfreeze in SigLIP")
    parser.add_argument("--enable-lora", action="store_true", default=True,
                        help="Enable LoRA fine-tuning (currently enabled by default in FusedVisionBackbone)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank")

    # ========== 融合模块参数 ==========
    parser.add_argument("--fusion-num-heads", type=int, default=4,
                        help="Number of attention heads in the fusion transformer")
    parser.add_argument("--fusion-num-layers", type=int, default=1,
                        help="Number of layers in the fusion transformer")

    # ========== 类别引导参数 ==========
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-k categories for prompt generation during inference")
    
    # ========== 损失函数超参数 ==========
    # InfoNCE Losses
    parser.add_argument("--tao-global", type=float, default=0.07,
                        help="Temperature for global InfoNCE loss")
    parser.add_argument("--tao-local", type=float, default=0.07,
                        help="Temperature for local (bit-level) InfoNCE loss")
    parser.add_argument("--hyper-lambda", type=float, default=1.0,
                        help="Scaling factor for all InfoNCE-based losses")
    parser.add_argument("--hyper-info-nce", type=float, default=5.0,
                        help="Weight for the global InfoNCE loss")
    parser.add_argument("--hyper-csq", type=float, default=5.0,
                        help="Weight for the csq loss")

    # Similarity-aware CE Loss
    parser.add_argument("--hyper-aware-ce-inter", type=float, default=5.0,
                        help="Weight for the similarity-aware-inter cross-entropy loss")
    parser.add_argument("--hyper-aware-ce-intra", type=float, default=0.005,
                        help="Weight for the similarity-aware-intra cross-entropy loss")
    # Quantization Loss
    parser.add_argument("--hyper-quan", type=float, default=1.0,
                        help="Weight for the quantization loss")
    parser.add_argument("--quant-std", type=float, default=0.5,
                        help="Standard deviation for BCE quantization loss")

    # Reconstruction Loss
    parser.add_argument("--mu", type=float, default=20.0,
                        help="Weight factor for auxiliary bit dimension losses")
    parser.add_argument("--hyper-recon", type=float, default=0.005,
                        help="Weight for the reconstruction loss")

    # Proxy Loss
    parser.add_argument("--hyper-proxy", type=float, default=2.0,
                        help="Weight for the proxy loss")

    # 分类损失
    parser.add_argument("--cls-loss-weight", type=float, default=0.2,
                        help="Weight for the auxiliary classification loss (Focal Loss)")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal loss alpha parameter")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")

    # ========== 实验参数 ==========
    parser.add_argument("--seed", type=int, default=3367,
                        help="Random seed")
    parser.add_argument("--rank", type=int, default=0,
                        help="GPU rank")

    # ========== 模式参数 ==========
    parser.add_argument("--is-train", action="store_true", default=True,
                        help="Set to training mode (default)")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Path to a pretrained model checkpoint")
    parser.add_argument("--result-name", type=str, default="result",
                        help="Base directory for saving results")
    
    # 解析一次以获取基本参数
    args, unknown = parser.parse_known_args()

    # 动态生成保存路径
    _time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.is_train:
        _time += "_test"

    k_bits_str = args.k_bits_list.replace(',', '_')
    save_dir = os.path.join(args.result_name, f"ScopeHash_{args.dataset}_{k_bits_str}bits", _time)
    
    # 将 save_dir 添加到 parser 的默认值中，然后重新解析
    parser.set_defaults(save_dir=save_dir)
    args = parser.parse_args()

    return args
