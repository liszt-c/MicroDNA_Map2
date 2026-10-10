#!/usr/bin/env python3
"""
benchmark/perturbation_test.py - 模型特征依赖性验证 (扰动测试)

  - Task 'infer': 加载模型权重，提取高分序列进行打乱和饱和突变推断，保存统计结果至 CSV。
  - Task 'plot': 读取已生成的 CSV 文件，绘制小提琴图(Violin Plot)和热图(Heatmap)。
  - Task 'all': 顺序执行上述两步。

使用示例:
  # 一次性执行推断与制图
  python benchmark/perturbation_test.py \
      --task all \
      --model models/best_model.pth \
      --input data/processed/eccDNA.fa

  # 分离执行 - 阶段1：仅推断并保存 CSV
  python benchmark/perturbation_test.py \
      --task infer \
      --model models/best_model.pth \
      --input data/processed/eccDNA.fa

  # 分离执行 - 阶段2：仅读取 CSV 绘图 (无需加载模型)
  python benchmark/perturbation_test.py \
      --task plot \
      --output-dir benchmark/results
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 确保能导入项目源码
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MODEL_DIR, DEFAULT_BATCH_SIZE, SEQUENCE_LENGTH
from src.model import ResNetSelfAttention
from src.dataprocess import encode_sequence, clean_sequence
from src.utils import setup_logger, iter_fasta_file

logger = setup_logger('perturbation_test')
BASES = ['A', 'C', 'G', 'T']


def load_model_weights(model_path: Path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = ResNetSelfAttention()
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def shuffle_sequence(seq: str) -> str:
    """保持碱基组分完全不变，随机打乱序列顺序"""
    seq_list = list(seq)
    random.shuffle(seq_list)
    return "".join(seq_list)


# =====================================================================
# Task: INFER (生成 CSV)
# =====================================================================

def infer_shuffling_test(model, fasta_path: Path, device, limit_n: int, output_csv: Path):
    """同组分序列打乱实验: 推断并保存 CSV"""
    logger.info("Starting GC-Preserved Shuffling Inference...")
    
    seqs = []
    for _, seq_raw in iter_fasta_file(fasta_path):
        seq = clean_sequence(seq_raw)
        if set(seq).issubset(set(BASES)):
            seqs.append(seq)
            if len(seqs) >= limit_n:
                break
                
    if not seqs:
        logger.error("No valid sequences found for shuffling test.")
        return

    shuffled_seqs = [shuffle_sequence(s) for s in seqs]
    
    def predict_batch(sequence_list):
        probs = []
        batch_size = 256
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence_list), batch_size), desc="Predicting", leave=False):
                batch_seqs = sequence_list[i:i+batch_size]
                tensors = [encode_sequence(s) for s in batch_seqs]
                batch_tensor = torch.from_numpy(np.array(tensors)).float().to(device)
                outputs = model(batch_tensor)
                batch_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                probs.extend(batch_prob)
        return probs

    logger.info("Scoring original sequences...")
    orig_probs = predict_batch(seqs)
    
    logger.info("Scoring shuffled sequences...")
    shuf_probs = predict_batch(shuffled_seqs)
    
    df = pd.DataFrame({
        'Original_Prob': orig_probs,
        'Shuffled_Prob': shuf_probs
    })
    df.to_csv(output_csv, index=False)
    logger.info(f"Shuffling inference complete. Data saved to {output_csv.name}")


def infer_mutagenesis_test(model, fasta_path: Path, device, limit_n: int, output_csv: Path):
    """在硅单碱基饱和突变实验: 推断并保存 CSV"""
    logger.info("Starting In Silico Saturation Mutagenesis Inference...")
    
    seqs = []
    for _, seq_raw in iter_fasta_file(fasta_path):
        seq = clean_sequence(seq_raw)
        if len(seq) >= 100 and set(seq).issubset(set(BASES)):
            seqs.append(seq)
            if len(seqs) >= limit_n:
                break
                
    if not seqs:
        logger.error("No valid sequences found for mutagenesis test.")
        return

    mutation_effects = {f"{o}->{m}": [] for o in BASES for m in BASES if o != m}
    
    for seq_idx, seq in enumerate(seqs):
        logger.info(f"Processing sequence {seq_idx+1}/{len(seqs)} for mutagenesis...")
        
        if len(seq) > SEQUENCE_LENGTH:
            start = (len(seq) - SEQUENCE_LENGTH) // 2
            seq = seq[start:start+SEQUENCE_LENGTH]
        elif len(seq) < SEQUENCE_LENGTH:
            continue
            
        with torch.no_grad():
            orig_tensor = torch.from_numpy(encode_sequence(seq)).float().unsqueeze(0).to(device)
            orig_prob = torch.softmax(model(orig_tensor), dim=1)[0, 1].item()
            
        if orig_prob < 0.8:
            continue
            
        mut_seqs = []
        mut_types = []
        for i in range(len(seq)):
            orig_base = seq[i]
            for mut_base in BASES:
                if mut_base != orig_base:
                    new_seq = seq[:i] + mut_base + seq[i+1:]
                    mut_seqs.append(new_seq)
                    mut_types.append(f"{orig_base}->{mut_base}")
                    
        batch_size = 512
        mut_probs = []
        with torch.no_grad():
            for i in range(0, len(mut_seqs), batch_size):
                b_seqs = mut_seqs[i:i+batch_size]
                b_tensors = [encode_sequence(s) for s in b_seqs]
                b_tensor = torch.from_numpy(np.array(b_tensors)).float().to(device)
                outputs = model(b_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                mut_probs.extend(probs)
                
        for m_type, m_prob in zip(mut_types, mut_probs):
            delta = m_prob - orig_prob
            mutation_effects[m_type].append(delta)

    avg_effects = {k: np.mean(v) if v else 0 for k, v in mutation_effects.items()}
    
    df = pd.DataFrame(list(avg_effects.items()), columns=['Mutation', 'Mean_Delta_Prob'])
    df.to_csv(output_csv, index=False)
    logger.info(f"Mutagenesis inference complete. Data saved to {output_csv.name}")


# =====================================================================
# Task: PLOT (读取 CSV 绘图)
# =====================================================================

def plot_shuffling_test(csv_path: Path, output_pdf: Path):
    """读取 Shuffling CSV 并绘制小提琴图"""
    if not csv_path.exists():
        logger.error(f"Cannot plot: {csv_path} does not exist.")
        return
        
    logger.info("Generating GC-Preserved Shuffling Plot...")
    df = pd.read_csv(csv_path)
    orig_probs = df['Original_Prob'].values
    shuf_probs = df['Shuffled_Prob'].values

    # 统计数据记录
    drop_ratio = (orig_probs > 0.5).sum() / max(len(orig_probs), 1)
    shuf_ratio = (shuf_probs > 0.5).sum() / max(len(shuf_probs), 1)
    logger.info(f"[Statistics] Original positive rate: {drop_ratio:.2%}, Shuffled positive rate: {shuf_ratio:.2%}")

    plt.figure(figsize=(8, 6))
    plot_data = pd.DataFrame({
        'Probability': np.concatenate([orig_probs, shuf_probs]),
        'Type': ['Original (Authentic)'] * len(orig_probs) + ['Shuffled (Identical Composition)'] * len(shuf_probs)
    })
    
    sns.violinplot(data=plot_data, x='Type', y='Probability', hue='Type', 
                   palette=['darkorange', 'steelblue'], inner="quartile", legend=False)
    
    plt.title("Impact of Sequence Randomization with Constant GC Content", fontsize=14)
    plt.ylabel("Predicted eccDNA Probability", fontsize=12)
    plt.xlabel("")
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf')
    plt.close()
    logger.info(f"Plot saved to {output_pdf.name}")


def plot_mutagenesis_test(csv_path: Path, output_pdf: Path):
    """读取 Mutagenesis CSV 并绘制热图"""
    if not csv_path.exists():
        logger.error(f"Cannot plot: {csv_path} does not exist.")
        return
        
    logger.info("Generating Saturation Mutagenesis Heatmap...")
    df = pd.read_csv(csv_path)
    
    # 构建 4x4 热图矩阵
    heatmap_data = np.zeros((4, 4))
    for i, orig in enumerate(BASES):
        for j, mut in enumerate(BASES):
            if orig != mut:
                mut_str = f"{orig}->{mut}"
                val_series = df.loc[df['Mutation'] == mut_str, 'Mean_Delta_Prob']
                val = val_series.values[0] if not val_series.empty else 0.0
                heatmap_data[i, j] = val
                
    plt.figure(figsize=(7, 6))
    sns.heatmap(heatmap_data, xticklabels=BASES, yticklabels=BASES, annot=True, 
                cmap="vlag", center=0, fmt=".4f", cbar_kws={'label': 'Mean $\Delta$ Probability'})
    plt.title("Effect of Single-Nucleotide Substitutions on Prediction", fontsize=13)
    plt.ylabel("Original Base", fontsize=12)
    plt.xlabel("Mutated Base", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf')
    plt.close()
    logger.info(f"Plot saved to {output_pdf.name}")


def main():
    p = argparse.ArgumentParser(description="MicroDNA Map Feature Dependency Benchmark")
    p.add_argument('--task', choices=['infer', 'plot', 'all'], default='all',
                   help="选择执行模式: infer(生成CSV), plot(读取CSV绘图), all(完整执行)")
    
    # 推断必需参数
    p.add_argument('--model', type=Path, default=None, help="Path to best_model.pth (required for infer)")
    p.add_argument('--input', type=Path, default=None, help="Path to positive FASTA (e.g., eccDNA.fa) (required for infer)")
    p.add_argument('--shuffling-n', type=int, default=5000, help="Number of sequences for shuffling test")
    p.add_argument('--mutagenesis-n', type=int, default=500, help="Number of sequences for mutagenesis test")
    
    # 统筹参数
    p.add_argument('--output-dir', type=Path, default=Path('benchmark/results'))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_shuffling = args.output_dir / "shuffling_test_results.csv"
    csv_mutagenesis = args.output_dir / "mutagenesis_results.csv"
    pdf_shuffling = args.output_dir / "shuffling_test_violin.pdf"
    pdf_mutagenesis = args.output_dir / "mutagenesis_heatmap.pdf"

    # ---- 阶段 1: 推断 (Infer) ----
    if args.task in ['infer', 'all']:
        if not args.model or not args.input:
            logger.error("Task 'infer' requires both --model and --input arguments.")
            sys.exit(1)
            
        model = load_model_weights(args.model, device)
        infer_shuffling_test(model, args.input, device, args.shuffling_n, csv_shuffling)
        infer_mutagenesis_test(model, args.input, device, args.mutagenesis_n, csv_mutagenesis)

    # ---- 阶段 2: 绘图 (Plot) ----
    if args.task in ['plot', 'all']:
        plot_shuffling_test(csv_shuffling, pdf_shuffling)
        plot_mutagenesis_test(csv_mutagenesis, pdf_mutagenesis)


if __name__ == "__main__":
    main()