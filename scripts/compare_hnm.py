#!/usr/bin/env python3
"""
scripts/compare_hnm.py - 评估困难负例挖掘(HNM)效果的对比分析工具

  - Task 'infer': 加载两个模型权重，在全量数据集上进行流式推理，提取 (Name, Label, Probability) 并保存为独立的 CSV。
  - Task 'plot': 读取已生成的 CSV 文件，使用核密度估计 (KDE) 绘制 2x2 概率分布图，直观展示拖尾消除及正例召回提升效果。
  - Task 'all': 顺序执行上述两步。
# 使用: 
方式一：一次性执行推断与制图
python scripts/compare_hnm.py \
    --task all \
    --model-base models/baseline_model.pth \
    --model-hnm models/hnm_model.pth

方式二：分离式执行 (推荐)
阶段 1: 仅执行推断并持久化保存 CSV, 便于其他统计软件复用排查顽固假阳性。
python scripts/compare_hnm.py \
    --task infer \
    --model-base models/baseline_model.pth \
    --model-hnm models/hnm_model.pth
阶段 2: 修改绘图参数后，无需重新推理，直接读取 CSV 高速出图。
python scripts/compare_hnm.py \
    --task plot \
    --csv-base results/metrics/hnm_comparison/Baseline_predictions.csv \
    --csv-hnm results/metrics/hnm_comparison/HNM_Model_predictions.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 确保能导入项目源码
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MODEL_DIR, METRICS_DIR, DEFAULT_BATCH_SIZE, NUM_WORKERS, PROCESSED_DATA_DIR
from src.model import ResNetSelfAttention
from src.dataloader import build_full_loader, ECC_LABEL
from src.utils import setup_logger

logger = setup_logger('compare_hnm')


def torch_load_compat(path: Path, device):
    """兼容不同 PyTorch 版本的模型加载"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: Path, device):
    """安全加载模型权重"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = ResNetSelfAttention()
    ckpt = torch_load_compat(model_path, device)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt.state_dict()

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def infer_and_save_csv(model, loader, dataset, device, output_csv: Path):
    """
    执行推理并将结果保存为 CSV。
    利用 DataLoader(shuffle=False) 保持与 dataset 索引的严格对齐。
    支持 CombinedMicroDNADataset 的索引解析。
    """
    logger.info(f"Starting inference... Saving to {output_csv.name}")
    
    names = []
    labels = []
    probs = []
    
    global_idx = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            
            # 获取属于 eccDNA (Class 1) 的概率
            batch_probs = torch.softmax(outputs, dim=1)[:, ECC_LABEL].cpu().numpy()
            batch_labels = targets.numpy()
            
            for i in range(len(batch_probs)):
                # 判断当前 dataset 是否为 CombinedMicroDNADataset
                if hasattr(dataset, 'n_ecc'):
                    if global_idx < dataset.n_ecc:
                        header = dataset.ecc_ds.headers[global_idx]
                    else:
                        header = dataset.other_ds.headers[global_idx - dataset.n_ecc]
                else:
                    header = dataset.headers[global_idx]
                
                names.append(header)
                labels.append(batch_labels[i])
                probs.append(batch_probs[i])
                global_idx += 1
                
    df = pd.DataFrame({
        'Name': names,
        'Label': labels,
        'Probability': probs
    })
    
    df['Class'] = df['Label'].map({0: 'otherDNA (Negative)', 1: 'eccDNA (Positive)'})
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"Inference complete. Saved {len(df)} records to {output_csv}")
    return df


def plot_kde_comparisons(df_base: pd.DataFrame, df_hnm: pd.DataFrame, output_pdf: Path, 
                         name_base: str, name_hnm: str):
    """
    读取 DataFrame，使用 KDE 绘制分离的分布密度图，并打印相关统计信息。
    """
    logger.info("Generating KDE distribution plots...")
    
    # --- 统计学占比计算与日志打印 ---
    # 提取正例和负例的概率
    pos_base = df_base[df_base['Label'] == 1]['Probability']
    neg_base = df_base[df_base['Label'] == 0]['Probability']
    pos_hnm = df_hnm[df_hnm['Label'] == 1]['Probability']
    neg_hnm = df_hnm[df_hnm['Label'] == 0]['Probability']

    # 正例统计
    logger.info("-" * 40)
    logger.info("Positive Samples (eccDNA) Analysis:")
    logger.info(f"  {name_base} Total Positives: {len(pos_base)}")
    logger.info(f"  {name_base} Positives > 0.5 (Recall): {(pos_base > 0.5).sum()} ({(pos_base > 0.5).mean():.2%})")
    logger.info(f"  {name_hnm} Positives > 0.5 (Recall): {(pos_hnm > 0.5).sum()} ({(pos_hnm > 0.5).mean():.2%})")
    
    # 负例统计
    logger.info("-" * 40)
    logger.info("Negative Samples (otherDNA) Tail Analysis:")
    logger.info(f"  {name_base} Total Negatives: {len(neg_base)}")
    logger.info(f"  {name_base} Negatives > 0.5 (False Pos): {(neg_base > 0.5).sum()} ({(neg_base > 0.5).mean():.2%})")
    logger.info(f"  {name_hnm} Negatives > 0.5 (False Pos): {(neg_hnm > 0.5).sum()} ({(neg_hnm > 0.5).mean():.2%})")
    logger.info("-" * 40)

    # --- 开始绘图 ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    color_neg = "steelblue"
    color_pos = "darkorange"

    # 子图 1 (0,0): Baseline 模型总体分布
    ax = axes[0, 0]
    sns.kdeplot(data=df_base[df_base['Label'] == 0], x='Probability', 
                fill=True, color=color_neg, label='Negative', ax=ax, clip=(0, 1))
    sns.kdeplot(data=df_base[df_base['Label'] == 1], x='Probability', 
                fill=True, color=color_pos, label='Positive', ax=ax, clip=(0, 1))
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_title(f"{name_base} Distribution")
    ax.set_xlim(0, 1)
    ax.legend()

    # 子图 2 (0,1): HNM 模型总体分布
    ax = axes[0, 1]
    sns.kdeplot(data=df_hnm[df_hnm['Label'] == 0], x='Probability', 
                fill=True, color=color_neg, label='Negative', ax=ax, clip=(0, 1))
    sns.kdeplot(data=df_hnm[df_hnm['Label'] == 1], x='Probability', 
                fill=True, color=color_pos, label='Positive', ax=ax, clip=(0, 1))
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_title(f"{name_hnm} Distribution")
    ax.set_xlim(0, 1)
    ax.legend()

    # 子图 3 (1,0): 对比负例的分布 (展示拖尾现象的消除)
    ax = axes[1, 0]
    sns.kdeplot(data=df_base[df_base['Label'] == 0], x='Probability', 
                fill=True, color="gray", alpha=0.3, label=f'{name_base} Negatives', ax=ax, clip=(0, 1))
    sns.kdeplot(data=df_hnm[df_hnm['Label'] == 0], x='Probability', 
                fill=False, color="red", linewidth=2, label=f'{name_hnm} Negatives', ax=ax, clip=(0, 1))
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_title("Negative Samples Tail Comparison (False Positives)")
    ax.set_xlim(0, 1)
    ax.legend()

    # 子图 4 (1,1): 对比正例的分布 (展示模型置信度的校准与召回提升)
    ax = axes[1, 1]
    sns.kdeplot(data=df_base[df_base['Label'] == 1], x='Probability', 
                fill=True, color="gray", alpha=0.3, label=f'{name_base} Positives', ax=ax, clip=(0, 1))
    sns.kdeplot(data=df_hnm[df_hnm['Label'] == 1], x='Probability', 
                fill=False, color="green", linewidth=2, label=f'{name_hnm} Positives', ax=ax, clip=(0, 1))
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_title("Positive Samples Comparison (Recall & Calibration)")
    ax.set_xlim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved successfully to {output_pdf}")


def main():
    p = argparse.ArgumentParser(description="Compare models with and without Hard Negative Mining")
    p.add_argument('--task', choices=['infer', 'plot', 'all'], default='all',
                   help="选择执行模式: infer(生成CSV), plot(读取CSV绘图), all(完整执行)")
    
    # 推理相关参数
    p.add_argument('--model-base', default=None, help="Baseline 模型权重路径 (无 HNM)")
    p.add_argument('--model-hnm', default=None, help="HNM 模型权重路径")
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    
    # 绘图相关及公用参数
    p.add_argument('--csv-base', type=Path, default=None, help="Baseline 推理结果 CSV 路径")
    p.add_argument('--csv-hnm', type=Path, default=None, help="HNM 推理结果 CSV 路径")
    p.add_argument('--name-base', default="Baseline", help="图表和文件名中 Baseline 模型的标签名称")
    p.add_argument('--name-hnm', default="HNM_Model", help="图表和文件名中 HNM 模型的标签名称")
    p.add_argument('--output-dir', type=Path, default=METRICS_DIR / "hnm_comparison")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 路径解析: 如果没有显式提供 csv 路径，则使用默认的输出目录及命名
    csv_base_path = args.csv_base if args.csv_base else args.output_dir / f"{args.name_base}_predictions.csv"
    csv_hnm_path = args.csv_hnm if args.csv_hnm else args.output_dir / f"{args.name_hnm}_predictions.csv"
    plot_out_path = args.output_dir / "hnm_density_comparison.pdf"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 任务 1: 推理 (Infer) ----
    if args.task in ['infer', 'all']:
        if not args.model_base or not args.model_hnm:
            logger.error("Task 'infer' requires both --model-base and --model-hnm to be provided.")
            sys.exit(1)
            
        logger.info("Executing Task: Inference")
        loader, dataset = build_full_loader(batch_size=args.batch_size, num_workers=args.num_workers)

        logger.info(f"Loading Baseline model: {args.model_base}")
        model_base = load_model(Path(args.model_base), device)
        infer_and_save_csv(model_base, loader, dataset, device, csv_base_path)
        del model_base
        torch.cuda.empty_cache()

        logger.info(f"Loading HNM model: {args.model_hnm}")
        model_hnm = load_model(Path(args.model_hnm), device)
        infer_and_save_csv(model_hnm, loader, dataset, device, csv_hnm_path)
        del model_hnm
        torch.cuda.empty_cache()

    # ---- 任务 2: 绘图 (Plot) ----
    if args.task in ['plot', 'all']:
        logger.info("Executing Task: Plotting")
        if not csv_base_path.exists() or not csv_hnm_path.exists():
            logger.error(f"Missing CSV files for plotting. Ensure they exist or run task 'infer' first.\n"
                         f"Checked: {csv_base_path} and {csv_hnm_path}")
            sys.exit(1)
            
        df_base = pd.read_csv(csv_base_path)
        df_hnm = pd.read_csv(csv_hnm_path)
        
        plot_kde_comparisons(df_base, df_hnm, plot_out_path, args.name_base, args.name_hnm)
        logger.info(f"Plot task completed. Outputs are in: {args.output_dir}")


if __name__ == "__main__":
    main()