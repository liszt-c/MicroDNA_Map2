#!/usr/bin/env python3
"""
scripts/sample_negatives.py - 从参考基因组中随机采样扩充负例库 (otherDNA)

"""
import argparse
import random
import sys
from pathlib import Path
import numpy as np

# 确保能正常导入上层目录的模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DATA_DIR, HG19_FA, SEQUENCE_LENGTH, SAMTOOLS
from src.utils import setup_logger, ensure_faidx, read_fai, extract_regions

logger = setup_logger('sample_negatives')


def count_fasta_records(path: Path) -> int:
    """快速统计 FASTA 文件中的序列条数"""
    if not path.exists():
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def is_valid_sequence(seq: str, target_len: int) -> bool:
    """检查序列是否全部由标准的 ACGT 组成, 剔除含有 N 的拼接区"""
    if len(seq) != target_len:
        return False
    # 只允许大写的 A, C, G, T
    return set(seq).issubset({'A', 'C', 'G', 'T'})


def main():
    p = argparse.ArgumentParser(
        description="Randomly sample negative DNA sequences (otherDNA) from reference genome.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--count', type=int, default=100000, 
                   help="要额外生成的负例数量 (如果不使用 --ratio)")
    p.add_argument('--ratio', type=float, default=None, 
                   help="目标正负比例 (负例总数 / 正例总数)。如果指定，将自动计算需要补充的负例数。")
    
    p.add_argument('--ecc-fasta', type=Path, default=PROCESSED_DATA_DIR / "eccDNA.fa",
                   help="正例数据文件路径 (用于统计正例数)")
    p.add_argument('--other-fasta', type=Path, default=PROCESSED_DATA_DIR / "otherDNA.fa",
                   help="负例数据文件路径 (用于统计现有负例，并将新数据追加到此文件)")
    p.add_argument('--ref', type=Path, default=HG19_FA,
                   help="参考基因组路径")
    p.add_argument('--target-len', type=int, default=SEQUENCE_LENGTH,
                   help="生成的序列长度 (bp)")
    p.add_argument('--batch-size', type=int, default=50000,
                   help="单次 samtools 查询的批处理量")
    
    args = p.parse_args()

    # 1. 统计当前的样本数量
    pos_count = count_fasta_records(args.ecc_fasta)
    neg_count = count_fasta_records(args.other_fasta)
    
    logger.info(f"Current dataset statistics:")
    logger.info(f"  Positive (eccDNA)  : {pos_count}")
    logger.info(f"  Negative (otherDNA): {neg_count}")

    # 2. 计算需要生成的负例数量
    if args.ratio is not None:
        if pos_count == 0:
            logger.error("Positive dataset is empty or missing. Cannot use --ratio. Please process eccDNA first.")
            sys.exit(1)
        target_neg_count = int(pos_count * args.ratio)
        to_generate = target_neg_count - neg_count
        
        logger.info(f"Target ratio set to {args.ratio}.")
        logger.info(f"Target total negative count: {target_neg_count}")
        
        if to_generate <= 0:
            logger.info("Current negative samples meet or exceed the target ratio. No new samples needed.")
            sys.exit(0)
    else:
        to_generate = args.count

    logger.info(f"Task: Generate {to_generate} new negative samples.")

    # 3. 解析参考基因组索引，确定采样范围
    ensure_faidx(args.ref, SAMTOOLS)
    fai = read_fai(args.ref)
    lengths = fai['lengths']
    
    if not lengths:
        logger.error(f"Failed to load reference index from {args.ref}.fai")
        sys.exit(1)

    # 过滤无效或者太短的染色体，仅保留常见的染色体和较大的 contigs
    valid_chroms = {}
    for chrom, length in lengths.items():
        # 排除长度不够以及可能具有特殊命名的碎片 (可选，这里用长度过滤保证基本有效)
        if length > args.target_len * 2 and ("chr" in chrom or chrom.isalnum()):
            valid_chroms[chrom] = length

    chrom_names = list(valid_chroms.keys())
    # 按染色体长度赋予被随机选中的权重 (实现全基因组均匀分布)
    chrom_probs = np.array(list(valid_chroms.values()), dtype=np.float64)
    chrom_probs /= chrom_probs.sum()

    # 4. 开始随机采样与提取
    args.other_fasta.parent.mkdir(parents=True, exist_ok=True)
    
    generated = 0
    batch_index = 1
    
    # 采用追加模式 'a' 写入 otherDNA.fa
    with open(args.other_fasta, 'a', encoding='utf-8') as out_f:
        while generated < to_generate:
            remaining = to_generate - generated
            # 因为部分区域会包含 N 被丢弃，多采样 20% 以防不足，上限不超过 batch_size
            current_batch_size = min(args.batch_size, int(remaining * 1.2))
            
            logger.info(f"Batch {batch_index}: Sampling {current_batch_size} candidate regions...")
            
            regions = []
            for _ in range(current_batch_size):
                chrom = np.random.choice(chrom_names, p=chrom_probs)
                # 1-based 坐标
                start = random.randint(1, valid_chroms[chrom] - args.target_len)
                # 确保取出的长度严格等于 target_len
                end = start + args.target_len - 1 
                regions.append((chrom, start, end))

            # 批量提取
            seq_map, skipped = extract_regions(args.ref, regions, SAMTOOLS)
            
            valid_count_in_batch = 0
            for chrom, start, end in regions:
                if generated >= to_generate:
                    break
                
                seq = seq_map.get((chrom, start, end), "").upper()
                
                if is_valid_sequence(seq, args.target_len):
                    # 组装格式化 Header, 添加 otherDNA 标签和原坐标信息
                    header = f">random_neg_{generated}_{chrom}:{start}-{end}|otherDNA"
                    out_f.write(f"{header}\n")
                    # 按 70 字符断行写入序列
                    for i in range(0, len(seq), 70):
                        out_f.write(seq[i:i+70] + "\n")
                    
                    generated += 1
                    valid_count_in_batch += 1
            
            logger.info(f"Batch {batch_index} done: Extracted {valid_count_in_batch} valid sequences. "
                        f"Progress: {generated}/{to_generate}")
            batch_index += 1

    logger.info("Successfully completed negative sample expansion.")
    logger.info(f"New negative samples added to: {args.other_fasta}")


if __name__ == "__main__":
    main()