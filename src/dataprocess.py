"""
src/dataprocess.py - 序列预处理与编码

"""
import random
import re
import numpy as np

from config import SEQUENCE_LENGTH

# 碱基 -> One-hot 行号 (与原版一致: A=0, T=1, C=2, G=3)
_BASE_TO_ROW = {ord('A'): 0, ord('T'): 1, ord('C'): 2, ord('G'): 3}


def clean_sequence(seq: str) -> str:
    """清理序列: 移除空白字符, 统一大写"""
    seq = seq.replace(" ", "").replace("\t", "") \
             .replace("\n", "").replace("\r", "")
    return seq.upper()


def encode_sequence(seq: str, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    DNA 序列 -> One-hot 矩阵 (4, length), dtype=float32

    - 序列 <= length: 居中放置, 两侧填充随机单碱基噪声 (与原版训练数据一致)
    - 序列 >  length: 截取正中间 length 个碱基
    - 非 ACGT 字符 (如 N) 对应位置全为 0
    """
    seq = clean_sequence(seq)
    n = len(seq)
    data = np.zeros((4, length), dtype=np.float32)

    if n == 0:
        # 全噪声填充
        rows = np.random.randint(0, 4, size=length)
        data[rows, np.arange(length)] = 1.0
        return data

    codes = np.frombuffer(seq.encode('ascii', errors='replace'), dtype=np.uint8).astype(np.int16)
    rows = np.full(n, -1, dtype=np.int16)
    for code, r in _BASE_TO_ROW.items():
        rows[codes == code] = r

    if n <= length:
        fund = (length - n) // 2

        # 左侧随机噪声
        if fund > 0:
            noise = np.random.randint(0, 4, size=fund)
            data[noise, np.arange(fund)] = 1.0

        # 真实序列 (居中)
        valid = rows >= 0
        if valid.any():
            cols = fund + np.nonzero(valid)[0]
            data[rows[valid], cols] = 1.0

        # 右侧随机噪声
        remaining = length - n - fund
        if remaining > 0:
            start = fund + n
            noise = np.random.randint(0, 4, size=remaining)
            data[noise, start + np.arange(remaining)] = 1.0
    else:
        # 截取正中间
        fund = (n - length) // 2
        sub_rows = rows[fund:fund + length]
        valid = sub_rows >= 0
        if valid.any():
            data[sub_rows[valid], np.nonzero(valid)[0]] = 1.0

    return data


def parse_fasta_header(header: str) -> dict:
    """
    极度鲁棒的 FASTA Header 解析器。
    
    支持格式:
      >SampleID|chr1:100-200|eccDNA      (process_data.py 生成的标准格式)
      >chr1:100-200                       (samtools 输出格式)
      >candidate_0_chr1:100-200           (cnvkit_pipeline 生成的格式)
      >chr13:52514242.0-52514642.0        (带异常浮点数)
    """
    info = {"id": "", "chrom": "", "start": 0, "end": 0,
            "has_position": False, "label": None}

    h = header.lstrip('>').strip()
    if not h:
        return info

    segments = [s.strip() for s in h.split('|') if s.strip()]

    # --- 解析位置: 完美兼容浮点数和小数点 .0 ---
    for seg in segments:
        match = re.search(r"([^:|\s]+):(\d+)(?:\.\d+)?-(\d+)(?:\.\d+)?", seg)
        if match:
            chrom_part = match.group(1).strip()
            # 若含 '_'，取最后一段 (candidate_0_chr1 -> chr1)
            if '_' in chrom_part:
                chrom_name = chrom_part.rsplit('_', 1)[-1]
            else:
                chrom_name = chrom_part
                
            info['chrom'] = chrom_name
            info['start'] = int(match.group(2))
            info['end'] = int(match.group(3))
            info['has_position'] = True
            break

    # --- 解析标签 ---
    for seg in segments[1:] if len(segments) > 1 else []:
        sl = seg.lower()
        if 'ecc' in sl:
            info['label'] = 1
            break
        if 'other' in sl or 'genomic' in sl:
            info['label'] = 0
            break

    # --- 解析 ID ---
    if len(segments) > 1:
        info['id'] = segments[0].split(':')[0].split()[0]
    else:
        info['id'] = h.split()[0].split(':')[0]

    return info