#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/pipeline/micro_coverage_pipeline.py - 微尺度局部覆盖度分析流程 (原生Python防广播断言版)

改进:
1. 彻底移除 extract_candidates 中的 Pandas 依赖，改用原生 Python 流式解析。
   杜绝底层 df.values 带来的多行内存广播错误（导致 196万行变成同一行的幽灵 BUG）。
2. 在 extract_candidates 加入探针，显式打印前两条候选样本的坐标以供核对。
3. 强化全息诊断引擎，无论提取是否丢弃序列，均输出报告确保绝对透明。
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pysam

from config import (
    BOWTIE2,
    BOWTIE2_BUILD,
    DEFAULT_MICRO_CLUSTER_MAX_LEN,
    DEFAULT_MICRO_EXEMPT_FOLD_CHANGE,
    DEFAULT_MICRO_FOLD_CHANGE,
    DEFAULT_MICRO_GC_CORRECTION,
    DEFAULT_MICRO_LOCAL_BASELINE_WINDOW,
    DEFAULT_MICRO_MAX_LEN,
    DEFAULT_MICRO_MIN_LEN,
    DEFAULT_MICRO_RELAX_RATIO,
    DEFAULT_MICRO_STEP_SIZE,
    DEFAULT_MICRO_WINDOW_SIZE,
    HG19_FA,
    MICRO_COVERAGE_TEMP_DIR,
    SAMTOOLS,
)
from ..utils import (
    ensure_bam_index,
    ensure_faidx,
    is_bam_valid,
    resolve_bowtie2_index,
    run_command,
    setup_logger,
)

logger = setup_logger("micro_coverage_pipeline")

CALL_CNS_REQUIRED_COLS = ("chromosome", "start", "end", "log2")


class MicroCoveragePipeline:
    """微尺度局部覆盖度分析类"""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        ref_genome: Optional[Path] = None,
        window_size: int = DEFAULT_MICRO_WINDOW_SIZE,
        step_size: int = DEFAULT_MICRO_STEP_SIZE,
        fold_change: float = DEFAULT_MICRO_FOLD_CHANGE,
        exempt_fold_change: float = DEFAULT_MICRO_EXEMPT_FOLD_CHANGE,
        relax_ratio: float = DEFAULT_MICRO_RELAX_RATIO,
        min_region_len: int = DEFAULT_MICRO_MIN_LEN,
        max_region_len: int = DEFAULT_MICRO_MAX_LEN,
        cluster_max_len: int = DEFAULT_MICRO_CLUSTER_MAX_LEN,
        flank_bins: int = 1,
        local_baseline_window: int = DEFAULT_MICRO_LOCAL_BASELINE_WINDOW,
        gc_correction: bool = DEFAULT_MICRO_GC_CORRECTION,
    ):
        self.output_dir = Path(output_dir) if output_dir else MICRO_COVERAGE_TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ref_genome = Path(ref_genome) if ref_genome else HG19_FA
        if not self.ref_genome.exists():
            raise FileNotFoundError(
                f"Reference genome not found: {self.ref_genome}\n"
                f"Please place reference genome into {self.ref_genome.parent}/"
            )

        prefix, _ = resolve_bowtie2_index(self.ref_genome)
        self.index_prefix = prefix
        self._index_checked = False

        ensure_faidx(self.ref_genome, SAMTOOLS)

        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.fold_change = float(fold_change)
        self.exempt_fold_change = float(exempt_fold_change)
        self.relax_ratio = float(relax_ratio)
        self.min_region_len = int(min_region_len)
        self.max_region_len = int(max_region_len)
        self.cluster_max_len = int(cluster_max_len)
        self.flank_bins = int(flank_bins)
        self.local_baseline_window = int(local_baseline_window)
        self.gc_correction = bool(gc_correction)

        self._gc_cache: Dict[str, np.ndarray] = {}

    def get_parameter_dict(self) -> Dict[str, str]:
        """返回关键分析参数字典，用于版本与缓存比对"""
        return {
            "window_size": str(self.window_size),
            "step_size": str(self.step_size),
            "fold_change": f"{self.fold_change:.4f}",
            "exempt_fold_change": f"{self.exempt_fold_change:.4f}",
            "cluster_max_len": str(self.cluster_max_len),
        }

    def build_index(self) -> None:
        """构建 Bowtie2 索引 (若缺失)"""
        prefix, exists = resolve_bowtie2_index(self.ref_genome)
        self.index_prefix = prefix
        if exists:
            logger.debug(f"Bowtie2 index already exists at {self.index_prefix}, skip building.")
            self._index_checked = True
            return

        logger.info(f"Building Bowtie2 index: {self.ref_genome} -> {self.index_prefix}")
        run_command([BOWTIE2_BUILD, "-f", str(self.ref_genome), str(self.index_prefix)])
        self._index_checked = True

    def align(
        self,
        fastq1: Path,
        fastq2: Path,
        sample_name: str,
        threads: int = 8,
        force_align: bool = False,
    ) -> Path:
        """FASTQ -> sorted & indexed BAM (支持复用已存在的有效 BAM 与索引)"""
        fastq1, fastq2 = Path(fastq1), Path(fastq2)
        bam_file = self.output_dir / f"{sample_name}.bam"

        if not force_align and is_bam_valid(bam_file, SAMTOOLS):
            logger.info(f"[{sample_name}] Found existing valid BAM: {bam_file.name}. Skipping alignment.")
            ensure_bam_index(bam_file, SAMTOOLS)
            return bam_file

        for fq in (fastq1, fastq2):
            if not fq.exists():
                raise FileNotFoundError(f"FASTQ not found: {fq}")

        if not self._index_checked:
            self.build_index()

        sam_file = self.output_dir / f"{sample_name}.sam"

        logger.info(f"[{sample_name}] bowtie2 aligning ({threads} threads) ...")
        run_command([
            BOWTIE2, "-p", str(threads),
            "-x", str(self.index_prefix),
            "-1", str(fastq1), "-2", str(fastq2),
            "-S", str(sam_file),
            "--no-unal",
        ])

        logger.info(f"[{sample_name}] samtools sort -> {bam_file.name} ...")
        run_command([SAMTOOLS, "sort", f"-@{threads}", "-o", str(bam_file), str(sam_file)])
        sam_file.unlink(missing_ok=True)

        logger.info(f"[{sample_name}] Ensuring BAM index for {bam_file.name} ...")
        ensure_bam_index(bam_file, SAMTOOLS)
        logger.info(f"[{sample_name}] BAM ready: {bam_file}")
        return bam_file

    def _get_chromosome_window_gc(self, chrom: str, chrom_len: int, num_windows: int) -> np.ndarray:
        if chrom in self._gc_cache:
            cached = self._gc_cache[chrom]
            if len(cached) == num_windows:
                return cached

        with pysam.FastaFile(str(self.ref_genome)) as fa:
            seq = fa.fetch(chrom, 0, chrom_len).upper()

        atomic_bins = int(math.ceil(chrom_len / self.step_size))
        atomic_gc = np.zeros(atomic_bins, dtype=np.int32)
        atomic_len = np.zeros(atomic_bins, dtype=np.int32)

        for i in range(atomic_bins):
            st = i * self.step_size
            en = min(st + self.step_size, chrom_len)
            sub = seq[st:en]
            atomic_len[i] = len(sub)
            if len(sub) > 0:
                atomic_gc[i] = sub.count("G") + sub.count("C")

        bins_per_win = max(1, self.window_size // self.step_size)
        if bins_per_win == 1:
            win_gc = atomic_gc[:num_windows]
            win_len = atomic_len[:num_windows]
        else:
            cumsum_gc = np.cumsum(np.insert(atomic_gc, 0, 0))
            cumsum_len = np.cumsum(np.insert(atomic_len, 0, 0))
            win_gc = (cumsum_gc[bins_per_win:] - cumsum_gc[:-bins_per_win])[:num_windows]
            win_len = (cumsum_len[bins_per_win:] - cumsum_len[:-bins_per_win])[:num_windows]

        gc_percent = np.zeros(num_windows, dtype=np.uint8)
        valid_mask = win_len > 0
        gc_percent[valid_mask] = np.round(100.0 * win_gc[valid_mask] / win_len[valid_mask]).astype(np.uint8)

        self._gc_cache[chrom] = gc_percent
        return gc_percent

    def _apply_gc_correction(
        self, window_counts: np.ndarray, gc_array: np.ndarray, global_baseline: float
    ) -> np.ndarray:
        if not self.gc_correction or global_baseline <= 0:
            return window_counts.astype(np.float32)

        norm_factors = np.ones(101, dtype=np.float32)
        non_zero = window_counts > 0

        for g in range(101):
            mask_g = (gc_array == g) & non_zero
            if np.sum(mask_g) >= 50:
                med = float(np.median(window_counts[mask_g]))
                if med > 0:
                    raw_factor = global_baseline / med
                    norm_factors[g] = float(np.clip(raw_factor, 0.33, 3.0))
                else:
                    norm_factors[g] = 1.0
            else:
                norm_factors[g] = 1.0

        corrected = window_counts.astype(np.float32) * norm_factors[gc_array]
        return corrected

    def _compute_local_baseline(
        self, counts: np.ndarray, global_baseline: float
    ) -> np.ndarray:
        window_pts = max(10, self.local_baseline_window // self.step_size)
        if len(counts) <= window_pts:
            return np.full_like(counts, fill_value=global_baseline, dtype=np.float32)

        pad_width = window_pts // 2
        padded = np.pad(counts, pad_width, mode="edge")
        cumsum = np.cumsum(np.insert(padded, 0, 0))
        rolling_sum = cumsum[window_pts:] - cumsum[:-window_pts]
        local_mean = rolling_sum[:len(counts)] / float(window_pts)

        clamped = np.clip(local_mean, 0.5 * global_baseline, 1.8 * global_baseline)
        return clamped.astype(np.float32)

    def _merge_candidate_regions(
        self, regions: List[Tuple[str, int, int, float]]
    ) -> List[Tuple[str, int, int, float]]:
        if not regions:
            return []

        sorted_regions = sorted(regions, key=lambda x: (x[0], x[1], x[2]))
        merged = []

        for chrom, s, e, l2 in sorted_regions:
            if not merged:
                merged.append([chrom, s, e, l2])
                continue

            last = merged[-1]
            if chrom == last[0] and s <= last[2]:
                last[2] = max(last[2], e)
                last[3] = max(last[3], l2)
            else:
                merged.append([chrom, s, e, l2])

        return [(m[0], m[1], m[2], m[3]) for m in merged]

    def call_cnv(
        self,
        bam_file: Path,
        sample_name: str,
        threads: int = 8,
        allowed_chroms: Optional[set] = None,
    ) -> Path:
        bam_file = Path(bam_file)
        if not bam_file.exists():
            raise FileNotFoundError(f"BAM file not found: {bam_file}")

        ensure_bam_index(bam_file, SAMTOOLS)
        call_out = self.output_dir / f"{sample_name}.call.cns"
        logger.info(
            f"[{sample_name}] Running enhanced micro-scale coverage profiling "
            f"(window={self.window_size}bp, step={self.step_size}bp, "
            f"fold_change={self.fold_change}, exempt_fc={self.exempt_fold_change})..."
        )

        raw_candidates: List[Tuple[str, int, int, float]] = []

        with pysam.AlignmentFile(str(bam_file), "rb") as bam:
            references = list(bam.references)
            lengths = list(bam.lengths)

            for chrom, chrom_len in zip(references, lengths):
                if allowed_chroms is not None and chrom not in allowed_chroms:
                    continue
                if chrom_len < self.window_size:
                    continue

                num_atomic_bins = int(math.ceil(chrom_len / self.step_size))
                atomic_counts = np.zeros(num_atomic_bins, dtype=np.uint32)

                try:
                    for read in bam.fetch(chrom):
                        if (
                            read.is_unmapped
                            or read.is_duplicate
                            or read.is_secondary
                            or read.mapping_quality < 20
                            or read.reference_start is None
                            or read.reference_end is None
                        ):
                            continue

                        span = read.reference_end - read.reference_start
                        if span <= 0 or span > 1000:
                            continue

                        s_bin = max(0, min(read.reference_start // self.step_size, num_atomic_bins - 1))
                        e_bin = max(0, min((read.reference_end - 1) // self.step_size, num_atomic_bins - 1))
                        atomic_counts[s_bin : e_bin + 1] += 1
                except Exception as e:
                    logger.warning(f"Error fetching reads on {chrom}: {e}")
                    continue

                bins_per_win = max(1, self.window_size // self.step_size)
                if num_atomic_bins < bins_per_win:
                    continue

                num_windows = num_atomic_bins - bins_per_win + 1
                cumsum_counts = np.cumsum(np.insert(atomic_counts, 0, 0))
                window_counts = (cumsum_counts[bins_per_win:] - cumsum_counts[:-bins_per_win])[:num_windows]

                non_zero = window_counts[window_counts > 0]
                if len(non_zero) == 0:
                    continue

                global_baseline = float(np.median(non_zero))
                global_baseline = max(global_baseline, 1.0)

                gc_array = self._get_chromosome_window_gc(chrom, chrom_len, num_windows)
                corrected_counts = self._apply_gc_correction(window_counts, gc_array, global_baseline)
                local_baseline = self._compute_local_baseline(corrected_counts, global_baseline)

                is_low_coverage = global_baseline < 5.0
                if is_low_coverage:
                    diff_req = 1.0
                    cutoff_swgs = np.maximum(2.0, local_baseline * max(1.5, self.fold_change))
                    enriched_mask = (corrected_counts >= cutoff_swgs) & ((corrected_counts - local_baseline) >= diff_req)
                else:
                    cutoff_main = local_baseline * self.fold_change
                    cutoff_exempt = local_baseline * self.exempt_fold_change
                    cutoff_relax = local_baseline * (1.0 + (self.fold_change - 1.0) * self.relax_ratio)
                    diff_req = np.maximum(3.0, 0.1 * local_baseline)

                    is_candidate = (corrected_counts >= cutoff_main) & ((corrected_counts - local_baseline) >= diff_req)
                    is_exempt = (corrected_counts >= cutoff_exempt) & ((corrected_counts - local_baseline) >= diff_req * 1.5)
                    is_neighbor = corrected_counts >= cutoff_relax

                    left_ok = np.pad(is_neighbor[:-1], (1, 0), constant_values=False)
                    right_ok = np.pad(is_neighbor[1:], (0, 1), constant_values=False)

                    enriched_mask = is_exempt | (is_candidate & (left_ok | right_ok))

                enriched_indices = np.where(enriched_mask)[0]
                if len(enriched_indices) == 0:
                    continue

                cur_start_idx = enriched_indices[0]
                cur_end_idx = enriched_indices[0] + 1

                for idx in enriched_indices[1:]:
                    if idx <= cur_end_idx + 1:
                        new_span = (idx - cur_start_idx) * self.step_size + self.window_size
                        if new_span <= self.cluster_max_len:
                            cur_end_idx = idx + 1
                            continue

                    reg_start = cur_start_idx * self.step_size
                    reg_end = min((cur_end_idx - 1) * self.step_size + self.window_size, chrom_len)
                    span = reg_end - reg_start

                    if self.min_region_len <= span <= self.cluster_max_len:
                        local_mean = float(np.mean(corrected_counts[cur_start_idx:cur_end_idx]))
                        mean_base = float(np.mean(local_baseline[cur_start_idx:cur_end_idx]))
                        log2_val = math.log2((local_mean + 1e-4) / max(mean_base, 1e-4))
                        raw_candidates.append((chrom, reg_start + 1, reg_end, log2_val))

                    cur_start_idx = idx
                    cur_end_idx = idx + 1

                reg_start = cur_start_idx * self.step_size
                reg_end = min((cur_end_idx - 1) * self.step_size + self.window_size, chrom_len)
                span = reg_end - reg_start
                if self.min_region_len <= span <= self.cluster_max_len:
                    local_mean = float(np.mean(corrected_counts[cur_start_idx:cur_end_idx]))
                    mean_base = float(np.mean(local_baseline[cur_start_idx:cur_end_idx]))
                    log2_val = math.log2((local_mean + 1e-4) / max(mean_base, 1e-4))
                    raw_candidates.append((chrom, reg_start + 1, reg_end, log2_val))

        final_candidates = self._merge_candidate_regions(raw_candidates)

        param_dict = self.get_parameter_dict()
        param_header_str = ",".join(f"{k}={v}" for k, v in param_dict.items())

        with open(call_out, "w", encoding="utf-8") as f:
            f.write(f"# params: {param_header_str}\n")
            f.write("chromosome\tstart\tend\tlog2\n")
            for c, s, e, l2 in final_candidates:
                f.write(f"{c}\t{s}\t{e}\t{l2:.4f}\n")

        logger.info(
            f"[{sample_name}] Micro-coverage profiling finished. "
            f"Identified {len(final_candidates)} candidate segment(s) -> {call_out.name}"
        )
        return call_out

    def align_and_call(
        self,
        fastq1: Path,
        fastq2: Path,
        sample_name: str,
        threads: int = 8,
        allowed_chroms: Optional[set] = None,
        force_align: bool = False,
    ) -> Path:
        bam = self.align(fastq1, fastq2, sample_name, threads=threads, force_align=force_align)
        return self.call_cnv(bam, sample_name, threads=threads, allowed_chroms=allowed_chroms)

    def extract_candidates(
        self,
        call_cns: Path,
        min_log2: Optional[float] = None,
        min_size: int = 0,
        max_size: Optional[int] = None,
        sample_name: Optional[str] = None,
    ) -> List[Dict]:
        call_cns = Path(call_cns)
        if not call_cns.exists():
            raise FileNotFoundError(f"Call file not found: {call_cns}")

        prefix = f"{sample_name}_" if sample_name else ""
        candidates = []
        n_total = 0

        # 🚀【核心修复】：完全放弃 Pandas 库的解析，改用原生 Python 逐行解析。
        # 彻底杜绝由于特殊长文本文件导致的 numpy.values 内存广播断层 BUG！
        with open(call_cns, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 过滤空行、参数行及表头
                if not line or line.startswith("#") or line.startswith("chromosome"):
                    continue
                
                parts = line.split()
                if len(parts) < 4:
                    continue
                    
                n_total += 1
                try:
                    chrom = str(parts[0]).strip()
                    start = int(float(parts[1]))
                    end = int(float(parts[2]))
                    log2_val = float(parts[3])
                except ValueError:
                    continue
                    
                size = end - start
                
                # 执行过滤逻辑
                if min_log2 is not None and log2_val < min_log2:
                    continue
                if min_size and size < min_size:
                    continue
                if max_size and size > max_size:
                    continue
                    
                candidates.append({
                    "name": f"{prefix}mcv{len(candidates)}",
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "log2": log2_val,
                    "size": size,
                })

        logger.info(
            f"[{call_cns.name}] Segments: {n_total} total -> {len(candidates)} after filtering "
            f"(min_log2={min_log2}, min_size={min_size}, max_size={max_size})"
        )
        
        # 【断言探针】：打印出解析结果的前2条，绝对确认数组没有变成完全一致的脏数据！
        if len(candidates) > 0:
            logger.info(f"Diagnostic - Sample 0: {candidates[0]['chrom']}:{candidates[0]['start']}-{candidates[0]['end']}")
            if len(candidates) > 1:
                logger.info(f"Diagnostic - Sample 1: {candidates[1]['chrom']}:{candidates[1]['start']}-{candidates[1]['end']}")
            
        return candidates

    def extract_sequences(self, candidates: List[Dict], output_fa: Path) -> int:
        output_fa = Path(output_fa)
        if not candidates:
            logger.warning("No candidates to extract; writing empty FASTA.")
            output_fa.write_text("")
            return 0

        ensure_faidx(self.ref_genome, SAMTOOLS)

        # 构建防复用集合 (使用原生遍历，确保每个独立的区间都被记录)
        uniq, seen = [], set()
        for c in candidates:
            key = (c["chrom"], c["start"], c["end"])
            if key not in seen:
                seen.add(key)
                uniq.append(c)

        logger.info(f"Unique candidate regions queued for extraction: {len(uniq)}")

        written = 0
        diag_unmapped = 0
        diag_out_of_bounds = 0
        diag_fetch_err = 0
        diag_empty = 0

        with pysam.FastaFile(str(self.ref_genome)) as fa, open(output_fa, "w", encoding="utf-8") as f:
            ref_chroms = list(fa.references)
            
            # 安全染色体映射字典
            chrom_map = {}
            for chrom in ref_chroms:
                chrom_map[chrom] = chrom
                chrom_map[chrom.lower()] = chrom
                chrom_map[chrom.upper()] = chrom
            
            for chrom in ref_chroms:
                cl = chrom.lower()
                if cl.startswith("chr"):
                    bare = chrom[3:]
                    if bare not in chrom_map: chrom_map[bare] = chrom
                    if bare.lower() not in chrom_map: chrom_map[bare.lower()] = chrom
                    if bare.upper() not in chrom_map: chrom_map[bare.upper()] = chrom
                else:
                    with_chr = f"chr{chrom}"
                    if with_chr not in chrom_map: chrom_map[with_chr] = chrom
                    if with_chr.lower() not in chrom_map: chrom_map[with_chr.lower()] = chrom
                    if with_chr.upper() not in chrom_map: chrom_map[with_chr.upper()] = chrom

            for c in uniq:
                raw_chrom = str(c["chrom"]).strip()
                ref_chrom = chrom_map.get(raw_chrom)
                if not ref_chrom:
                    diag_unmapped += 1
                    continue

                chrom_len = fa.get_reference_length(ref_chrom)
                start = max(1, int(c["start"]))
                end = min(int(c["end"]), chrom_len)
                
                if start > end or start > chrom_len:
                    diag_out_of_bounds += 1
                    continue

                try:
                    seq = fa.fetch(ref_chrom, start - 1, end).upper()
                except Exception as e:
                    diag_fetch_err += 1
                    continue

                if not seq:
                    diag_empty += 1
                    continue

                f.write(f">{c['name']}|{ref_chrom}:{start}-{end}\n")
                for i in range(0, len(seq), 70):
                    f.write(seq[i : i + 70] + "\n")
                written += 1

        logger.info(
            f"Extraction Diagnostics -> Total unique: {len(uniq)} | "
            f"Written: {written} | "
            f"Unmapped chroms: {diag_unmapped} | "
            f"Out of bounds: {diag_out_of_bounds} | "
            f"Fetch errors: {diag_fetch_err} | "
            f"Empty strings: {diag_empty}"
        )
        
        return written

    @staticmethod
    def candidates_to_bed_rows(candidates: List[Dict]) -> List[Tuple[str, int, int]]:
        rows = []
        for c in candidates:
            rows.append((c["chrom"], max(0, int(c["start"]) - 1), int(c["end"])))
        return rows

    def cleanup_sample(self, sample_name: str, keep_bam: bool = False) -> int:
        patterns = [
            f"{sample_name}.sam",
            f"{sample_name}_candidates.fa",
        ]
        if not keep_bam:
            patterns += [
                f"{sample_name}.bam",
                f"{sample_name}.bam.bai",
                f"{sample_name}.bai",
            ]

        removed = 0
        for pat in patterns:
            for p in self.output_dir.glob(pat):
                try:
                    p.unlink()
                    removed += 1
                except OSError as e:
                    logger.warning(f"Failed to remove {p}: {e}")
        logger.info(f"[{sample_name}] Removed {removed} intermediate files.")
        return removed