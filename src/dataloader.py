"""
src/dataloader.py - PyTorch Dataset 与染色体平衡 5 折交叉验证分流器
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from config import (
    CHROM_FOLDS,
    DEFAULT_BATCH_SIZE,
    NUM_CV_FOLDS,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from .dataprocess import encode_sequence, parse_fasta_header
from .utils import setup_logger

logger = setup_logger('dataloader')

ECC_LABEL = 1        # eccDNA (Positive)
OTHER_LABEL = 0      # otherDNA / genomic background (Negative)


def normalize_chrom_name(chrom: str) -> str:
    """标准化染色体名称，消除大小写与 'chr' 前缀差异"""
    if not chrom:
        return ""
    c = str(chrom).strip()
    cl = c.lower()
    suffix = cl[3:] if cl.startswith("chr") else cl
    
    if suffix in [str(i) for i in range(1, 23)]:
        return f"chr{suffix}"
    elif suffix in ["x", "y"]:
        return f"chr{suffix.upper()}"
    elif suffix in ["m", "mt"]:
        return "chrM"
    return f"chr{suffix}" if not c.startswith("chr") else c


# 构建染色体 -> Fold 映射查找表
_CHROM_TO_FOLD_MAP: Dict[str, int] = {}
for _f_idx, _c_list in enumerate(CHROM_FOLDS):
    for _c in _c_list:
        _norm = normalize_chrom_name(_c)
        _CHROM_TO_FOLD_MAP[_norm] = _f_idx
        _bare = _norm[3:] if _norm.startswith("chr") else _norm
        _CHROM_TO_FOLD_MAP[_bare] = _f_idx


def get_chrom_fold(chrom: str) -> int:
    """获取指定染色体归属的 Fold 索引 (0 到 NUM_CV_FOLDS-1)"""
    norm = normalize_chrom_name(chrom)
    if norm in _CHROM_TO_FOLD_MAP:
        return _CHROM_TO_FOLD_MAP[norm]
    bare = norm[3:] if norm.startswith("chr") else norm
    if bare in _CHROM_TO_FOLD_MAP:
        return _CHROM_TO_FOLD_MAP[bare]
    # 对未命名的非典型 contig 采用哈希确定性分流，绝不丢失数据
    return abs(hash(norm)) % NUM_CV_FOLDS


def seed_worker(worker_id: int):
    """DataLoader worker 初始化: 设置独立的随机种子"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MicroDNADataset(Dataset):
    """从 FASTA 文件以惰性二进制寻道方式加载序列，并记录染色体属性"""

    def __init__(self, fasta_path, fallback_label: int = None):
        self.fasta_path = Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")

        if fallback_label is None:
            fallback_label = ECC_LABEL if 'eccdna' in self.fasta_path.name.lower() else OTHER_LABEL
        self.fallback_label = int(fallback_label)

        self.headers: List[str] = []
        self.offsets: List[int] = []
        self.labels: List[int] = []
        self.chroms: List[str] = []

        self._fh = None
        self._fh_pid = None

        self._build_index()
        self.size = len(self.offsets)
        if self.size == 0:
            raise ValueError(f"No FASTA records found in {self.fasta_path}")

        n_ecc = sum(1 for l in self.labels if l == ECC_LABEL)
        logger.info(f"Indexed {self.size} records from {self.fasta_path.name} (ecc={n_ecc}, other={self.size - n_ecc})")

    def _build_index(self):
        with open(self.fasta_path, 'rb') as f:
            pos = f.tell()
            line = f.readline()
            while line:
                if line.startswith(b'>'):
                    header = line.decode('utf-8', errors='replace').strip()
                    info = parse_fasta_header(header)
                    label = info['label'] if info['label'] is not None else self.fallback_label
                    chrom = normalize_chrom_name(info['chrom']) if info['chrom'] else ""
                    self.headers.append(header)
                    self.offsets.append(pos)
                    self.labels.append(int(label))
                    self.chroms.append(chrom)
                pos = f.tell()
                line = f.readline()

    def _get_fh(self):
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(self.fasta_path, 'rb')
            self._fh_pid = pid
        return self._fh

    def _read_sequence(self, idx: int) -> str:
        fh = self._get_fh()
        fh.seek(self.offsets[idx])
        fh.readline()
        parts = []
        for line in fh:
            if line.startswith(b'>'):
                break
            parts.append(line.strip())
        return b''.join(parts).decode('utf-8', errors='replace')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"index {idx} out of range [0, {self.size})")
        seq = self._read_sequence(idx)
        data = torch.from_numpy(encode_sequence(seq)).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

    def get_header_info(self, idx: int) -> dict:
        return parse_fasta_header(self.headers[idx])

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_fh'] = None
        state['_fh_pid'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


class CombinedMicroDNADataset(Dataset):
    """联合正例 (eccDNA.fa) 与负例 (otherDNA.fa)"""

    def __init__(self, ecc_path=None, other_path=None):
        ecc_path = Path(ecc_path) if ecc_path else PROCESSED_DATA_DIR / "eccDNA.fa"
        other_path = Path(other_path) if other_path else PROCESSED_DATA_DIR / "otherDNA.fa"
        if not ecc_path.exists() or not other_path.exists():
            raise FileNotFoundError(
                f"Processed FASTA not found (expected {ecc_path} and {other_path}). "
                f"Please run scripts/process_data.py and sample_negatives.py first."
            )
        self.ecc_ds = MicroDNADataset(ecc_path, fallback_label=ECC_LABEL)
        self.other_ds = MicroDNADataset(other_path, fallback_label=OTHER_LABEL)
        self.n_ecc = len(self.ecc_ds)

    def __len__(self):
        return self.n_ecc + len(self.other_ds)

    def __getitem__(self, idx):
        if idx < self.n_ecc:
            return self.ecc_ds[idx]
        return self.other_ds[idx - self.n_ecc]

    def label_of(self, idx: int) -> int:
        return ECC_LABEL if idx < self.n_ecc else OTHER_LABEL

    def chrom_of(self, idx: int) -> str:
        if idx < self.n_ecc:
            return self.ecc_ds.chroms[idx]
        return self.other_ds.chroms[idx - self.n_ecc]


def subset_labels(full_ds: CombinedMicroDNADataset, subset: Subset) -> np.ndarray:
    """提取子集样本标签数组"""
    return np.array([full_ds.label_of(i) for i in subset.indices], dtype=np.int64)


def build_chromosome_cv_loaders(
    test_fold: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = RANDOM_SEED,
    num_workers: int = NUM_WORKERS,
    balanced: bool = False,
    ecc_path=None,
    other_path=None,
) -> Tuple[DataLoader, DataLoader, DataLoader, CombinedMicroDNADataset, Subset, Subset, Subset]:
    """
    构建平衡型染色体 5 折交叉验证数据加载器。
    
    分区协议:
      - Test Fold: fold_idx = test_fold % 5 (完全独立的测试染色体)
      - Val Fold:  fold_idx = (test_fold + 1) % 5 (验证与早停染色体)
      - Train Folds: 其余 3 个 Fold 的全部染色体
    
    :param test_fold: 当前作为测试集的 Fold 索引 (0 到 4)
    :param balanced: 训练集是否启用按类加权平衡采样
    """
    dataset = CombinedMicroDNADataset(ecc_path, other_path)
    total_samples = len(dataset)

    # 1. 扫描全量样本，按染色体硬分配至 5 个 Fold
    fold_indices: List[List[int]] = [[] for _ in range(NUM_CV_FOLDS)]
    for idx in range(total_samples):
        chrom = dataset.chrom_of(idx)
        f_id = get_chrom_fold(chrom)
        fold_indices[f_id].append(idx)

    # 2. 划分当前迭代的角色 Fold
    test_idx = test_fold % NUM_CV_FOLDS
    val_idx = (test_fold + 1) % NUM_CV_FOLDS
    train_fold_ids = [f for f in range(NUM_CV_FOLDS) if f != test_idx and f != val_idx]

    test_indices = fold_indices[test_idx]
    val_indices = fold_indices[val_idx]
    train_indices: List[int] = []
    for f in train_fold_ids:
        train_indices.extend(fold_indices[f])

    # 乱序打乱训练集样本索引
    rng = np.random.default_rng(seed + test_fold)
    rng.shuffle(train_indices)

    train_sub = Subset(dataset, train_indices)
    val_sub = Subset(dataset, val_indices)
    test_sub = Subset(dataset, test_indices)

    # 3. 训练集类别平衡采样 (WeightedRandomSampler)
    sampler = None
    if balanced:
        labels = subset_labels(dataset, train_sub)
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        if (counts == 0).any():
            raise ValueError(f"Balanced sampling impossible, class counts = {counts.tolist()}")
        weight_per_class = 1.0 / counts
        weights = weight_per_class[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    g = torch.Generator().manual_seed(seed + test_fold)
    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=g, drop_last=False
    )
    val_loader = DataLoader(
        val_sub, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=g, drop_last=False
    )
    test_loader = DataLoader(
        test_sub, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=g, drop_last=False
    )

    logger.info(
        f"[CV Fold {test_fold + 1}/{NUM_CV_FOLDS}] Partition Summary:\n"
        f"  * Test Fold  ({test_idx + 1}): {CHROM_FOLDS[test_idx]} | {len(test_sub)} samples\n"
        f"  * Val Fold   ({val_idx + 1}): {CHROM_FOLDS[val_idx]} | {len(val_sub)} samples\n"
        f"  * Train Folds ({[f + 1 for f in train_fold_ids]}): {len(train_sub)} samples"
    )
    return train_loader, val_loader, test_loader, dataset, train_sub, val_sub, test_sub


def build_train_val_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split: float = 0.2,
    seed: int = RANDOM_SEED,
    num_workers: int = NUM_WORKERS,
    balanced: bool = False,
    ecc_path=None,
    other_path=None,
):
    """经典随机划分 DataLoader 构建函数 (保留向后兼容性)"""
    dataset = CombinedMicroDNADataset(ecc_path, other_path)
    total = len(dataset)
    val_n = int(round(total * val_split))
    train_n = total - val_n

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=g)

    sampler = None
    if balanced:
        labels = subset_labels(dataset, train_ds)
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        weight_per_class = 1.0 / np.maximum(counts, 1)
        weights = weight_per_class[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=g, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=g, drop_last=False
    )
    return train_loader, val_loader, dataset, train_ds, val_ds


def build_full_loader(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    ecc_path=None,
    other_path=None,
):
    """构建全量数据加载器 (无打乱，用于全景推断与测试)"""
    dataset = CombinedMicroDNADataset(ecc_path, other_path)
    g = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
    return loader, dataset