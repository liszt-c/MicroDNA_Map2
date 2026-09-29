"""
scripts/predict.py - 高吞吐量模型推理引擎

特性:
  1. 支持单模型推断与 5 折交叉验证权重集成软投票 (Ensemble Soft-Voting)
  2. 提供短序列批处理 (short) 与长序列双层滑动窗口扫描 (long)
  3. 补全 predict_long_single 接口，保证 batch_process.py 一次通过
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    MODEL_DIR, PREDICTIONS_DIR, SEQUENCE_LENGTH,
    SLIDE_STEP1, SLIDE_WINDOW2, SLIDE_STEP2, CV_MODEL_DIR
)
from src.model import ResNetSelfAttention
from src.dataprocess import encode_sequence, clean_sequence, parse_fasta_header
from src.utils import setup_logger, iter_fasta_file

logger = setup_logger('predict')

STEP1 = SLIDE_STEP1
WINDOW2 = SLIDE_WINDOW2
STEP2 = SLIDE_STEP2


def torch_load_compat(path: Path, device):
    """跨版本 PyTorch 权重加载"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class EnsembleModel(torch.nn.Module):
    """5 折交叉验证模型集成容器 (Soft-Voting)"""
    def __init__(self, models: List[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对 5 个模型的 Softmax 概率进行平均
        probs = [torch.softmax(m(x), dim=1) for m in self.models]
        avg_probs = torch.stack(probs, dim=0).mean(dim=0)
        # 转换为对数空间输出，保证与后续下游 softmax 接口无缝契合
        return torch.log(avg_probs + 1e-9)


def load_single_model(model_path: Path, device) -> Tuple[torch.nn.Module, int]:
    """加载单个模型权重"""
    ckpt = torch_load_compat(model_path, device)
    ecc_class_index = 1
    layer_size = 8

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        if 'layer_size' in ckpt:
            layer_size = ckpt['layer_size']
        conv = str(ckpt.get('label_convention', ''))
        if 'eccdna=0' in conv.lower():
            ecc_class_index = 0
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        ckpt.to(device)
        ckpt.eval()
        return ckpt, 0

    if 'shared_layers.0.weight' in state:
        layer_size = state['shared_layers.0.weight'].shape[0]

    model = ResNetSelfAttention(layer_size=layer_size)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, ecc_class_index


def load_model(model_path_or_dir: Union[str, Path], device) -> Tuple[torch.nn.Module, int]:
    """
    智能模型加载器:
    - 若传入目录且包含多个 fold 权重，自动组装为 5 折集成模型 (EnsembleModel)
    - 若传入单个文件，则加载常规单一模型
    """
    target = Path(model_path_or_dir)
    if not target.exists():
        raise FileNotFoundError(f"Model path does not exist: {target}")

    if target.is_dir():
        pth_files = sorted(list(target.glob("fold*_best_model.pth")) or list(target.glob("best_model_fold*.pth")) or list(target.glob("*.pth")))
        if not pth_files:
            raise FileNotFoundError(f"No .pth weights found in directory: {target}")
        if len(pth_files) == 1:
            logger.info(f"Loading single checkpoint from directory: {pth_files[0].name}")
            return load_single_model(pth_files[0], device)

        logger.info(f"Assembling 5-Fold Ensemble Model from {len(pth_files)} checkpoints in {target.name}...")
        models = []
        ecc_idx = 1
        for p in pth_files:
            m, ecc_idx = load_single_model(p, device)
            models.append(m)
        ensemble = EnsembleModel(models)
        ensemble.to(device)
        ensemble.eval()
        logger.info(f"Ensemble successfully constructed ({len(models)} models).")
        return ensemble, ecc_idx
    else:
        return load_single_model(target, device)


def merge_overlapping_regions(regions: List[Tuple[int, int]], gap_tolerance: int = 0) -> List[Tuple[int, int]]:
    """合并重叠或相邻的候选区间"""
    if not regions:
        return []
    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged = [sorted_regions[0]]
    for start, end in sorted_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap_tolerance:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _write_fasta_record(ff, hdr: str, seq: str, line_width: int = 70):
    ff.write(hdr + "\n")
    for i in range(0, len(seq), line_width):
        ff.write(seq[i : i + line_width] + "\n")


class SequenceTracker:
    """跟踪长序列在滑动窗口下的概率图谱"""
    def __init__(self, rec_idx: int, header: str, seq_raw: str):
        self.rec_idx = rec_idx
        self.header = header
        self.seq_clean = clean_sequence(seq_raw)
        self.n = len(self.seq_clean)
        self.chrom_info = parse_fasta_header(header)
        
        if self.n >= SEQUENCE_LENGTH:
            self.num_windows = (self.n - SEQUENCE_LENGTH) // STEP1 + 1
        else:
            self.num_windows = 1

        self.probs = np.zeros(self.num_windows, dtype=np.float32)
        self.filled_windows = 0

    def get_window_seq(self, win_idx: int) -> np.ndarray:
        if self.n >= SEQUENCE_LENGTH:
            start = win_idx * STEP1
            end = start + SEQUENCE_LENGTH
            return encode_sequence(self.seq_clean[start:end])
        return encode_sequence(self.seq_clean)

    def is_complete(self) -> bool:
        return self.filled_windows >= self.num_windows


def finalize_sequence_regions(tracker: SequenceTracker, limit: float, min_region_len: int, do_merge: bool):
    """对原始概率图谱应用二级滑动窗口平滑与阈值离散化"""
    n = tracker.n
    probs = tracker.probs

    if n < SEQUENCE_LENGTH:
        if probs[0] >= limit:
            return [(0, n)], tracker.seq_clean, tracker.chrom_info
        return [], tracker.seq_clean, tracker.chrom_info

    if len(probs) < WINDOW2:
        smoothed = probs.copy()
    else:
        num_win2 = (len(probs) - WINDOW2) // STEP2 + 1
        smoothed = np.zeros(num_win2, dtype=np.float32)
        for j in range(num_win2):
            s = j * STEP2
            e = s + WINDOW2
            smoothed[j] = np.mean(probs[s:e])

    binary = (smoothed >= limit).astype(np.int8)
    raw_regions = []
    in_region = False
    left = 0
    for idx, val in enumerate(binary):
        if val == 1 and not in_region:
            in_region = True
            left = idx
        elif val == 0 and in_region:
            in_region = False
            raw_regions.append((left, idx))
    if in_region:
        raw_regions.append((left, len(binary)))

    final_regions = []
    for l, r in raw_regions:
        win_start_idx = l * STEP2
        win_end_idx = (r - 1) * STEP2
        seq_start = max(0, win_start_idx * STEP1)
        seq_end = min(n, win_end_idx * STEP1 + SEQUENCE_LENGTH)
        if (seq_end - seq_start) >= min_region_len:
            final_regions.append((seq_start, seq_end))

    if do_merge:
        final_regions = merge_overlapping_regions(final_regions, gap_tolerance=0)

    return final_regions, tracker.seq_clean, tracker.chrom_info


def predict_long_single(
    header: str,
    seq_raw: str,
    model: torch.nn.Module,
    device,
    ecc_class_index: int = 1,
    limit: float = 0.75,
    batch_size: int = 512,
    min_region_len: int = 150,
    do_merge: bool = True,
) -> Tuple[List[Tuple[int, int]], str, dict]:
    """单条长序列预测接口，供 batch_process.py 直接调用"""
    tracker = SequenceTracker(0, header, seq_raw)
    if tracker.num_windows == 0:
        return [], tracker.seq_clean, tracker.chrom_info

    all_windows = [tracker.get_window_seq(w) for w in range(tracker.num_windows)]
    probs_list = []

    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            chunk = all_windows[i : i + batch_size]
            tensor = torch.from_numpy(np.array(chunk)).float().to(device)
            outputs = model(tensor)
            chunk_probs = torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy()
            probs_list.extend(chunk_probs)

    tracker.probs = np.array(probs_list, dtype=np.float32)
    tracker.filled_windows = tracker.num_windows

    regions, seq_clean, chrom_info = finalize_sequence_regions(
        tracker, limit=limit, min_region_len=min_region_len, do_merge=do_merge
    )
    return regions, seq_clean, chrom_info


def process_short_batched(fa_file: Path, output_dir: Path, model, device, ecc_class_index: int, batch_size: int):
    tsv_path = output_dir / f"{fa_file.stem}_short_results.tsv"
    n_rec = 0
    batch_headers, batch_seqs = [], []

    with open(tsv_path, 'w', encoding='utf-8') as tf:
        tf.write("header\teccDNA_prob\n")
        for header, seq in iter_fasta_file(fa_file):
            batch_headers.append(header)
            batch_seqs.append(encode_sequence(seq))

            if len(batch_seqs) >= batch_size:
                tensor = torch.from_numpy(np.array(batch_seqs)).float().to(device)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy()
                for h, p in zip(batch_headers, probs):
                    tf.write(f"{h}\t{p:.6f}\n")
                n_rec += len(batch_headers)
                batch_headers, batch_seqs = [], []

        if batch_seqs:
            tensor = torch.from_numpy(np.array(batch_seqs)).float().to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy()
            for h, p in zip(batch_headers, probs):
                tf.write(f"{h}\t{p:.6f}\n")
            n_rec += len(batch_headers)

    logger.info(f"Short results ({n_rec} records) -> {tsv_path}")


def process_long_batched(fa_file: Path, output_dir: Path, model, device, ecc_class_index: int, args):
    bed_path = output_dir / f"{fa_file.stem}.bed"
    fasta_path = output_dir / f"{fa_file.stem}_candidates.fasta"
    do_merge = not args.no_merge
    cand_counter = 0

    window_batch_tensors = []
    window_batch_routes = []
    active_trackers = {}
    rec_counter = 0

    with open(bed_path, 'w', encoding='utf-8') as bf, open(fasta_path, 'w', encoding='utf-8') as ff:
        def flush_batch():
            nonlocal cand_counter
            if not window_batch_tensors:
                return
            tensor = torch.from_numpy(np.array(window_batch_tensors)).float().to(device)
            with torch.no_grad():
                outputs = model(tensor)
                batch_probs = torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy()

            for (t_id, w_idx), prob in zip(window_batch_routes, batch_probs):
                tracker = active_trackers[t_id]
                tracker.probs[w_idx] = prob
                tracker.filled_windows += 1

            window_batch_tensors.clear()
            window_batch_routes.clear()

            completed_ids = [t_id for t_id, tr in active_trackers.items() if tr.is_complete()]
            for t_id in completed_ids:
                tracker = active_trackers.pop(t_id)
                regions, seq_clean, chrom_info = finalize_sequence_regions(
                    tracker, limit=args.limit, min_region_len=args.min_region_len, do_merge=do_merge
                )
                chrom = chrom_info.get('chrom', '') or f"seq{tracker.rec_idx}"
                base_offset = max(0, chrom_info.get('start', 1) - 1) if chrom_info.get('has_position', False) else 0

                for rs, re in regions:
                    cand_counter += 1
                    abs_s = base_offset + rs
                    abs_e = base_offset + re
                    bf.write(f"{chrom}\t{max(0, abs_s)}\t{abs_e}\n")
                    fasta_hdr = f">candidate_{cand_counter}_{chrom}:{abs_s}-{abs_e}"
                    _write_fasta_record(ff, fasta_hdr, seq_clean[rs:re])

                bf.flush()
                ff.flush()

        pbar = tqdm(desc=f"Streaming {fa_file.name}", unit="seq")
        for header, seq_raw in iter_fasta_file(fa_file):
            tracker = SequenceTracker(rec_counter, header, seq_raw)
            active_trackers[rec_counter] = tracker

            for w in range(tracker.num_windows):
                window_batch_tensors.append(tracker.get_window_seq(w))
                window_batch_routes.append((rec_counter, w))
                if len(window_batch_tensors) >= args.batch_size:
                    flush_batch()

            rec_counter += 1
            pbar.update(1)
        pbar.close()

        while active_trackers:
            flush_batch()

    logger.info(f"{fa_file.name}: {cand_counter} candidate(s) logged -> {bed_path}")


def main():
    p = argparse.ArgumentParser(description="MicroDNA High-Throughput Prediction Engine")
    p.add_argument('--input', required=True, help="FASTA 文件或目录")
    p.add_argument('--mode', choices=['short', 'long'], default='long')
    p.add_argument('--model', default=None, help="模型路径 (.pth 或包含 5 折权重的目录)")
    p.add_argument('--limit', type=float, default=0.75)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--min-region-len', type=int, default=150)
    p.add_argument('--no-merge', action='store_true')
    p.add_argument('--output-dir', default=str(PREDICTIONS_DIR))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_target = Path(args.model) if args.model else (CV_MODEL_DIR if CV_MODEL_DIR.exists() else MODEL_DIR / "best_model.pth")
    model, ecc_class_index = load_model(model_target, device)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fa_files = [input_path] if input_path.is_file() else sorted(list(input_path.glob("*.fa")) + list(input_path.glob("*.fasta")))
    for fa_file in fa_files:
        logger.info(f"Processing {fa_file.name} [Mode: {args.mode}] ...")
        if args.mode == 'short':
            process_short_batched(fa_file, output_dir, model, device, ecc_class_index, args.batch_size)
        else:
            process_long_batched(fa_file, output_dir, model, device, ecc_class_index, args)


if __name__ == '__main__':
    main()