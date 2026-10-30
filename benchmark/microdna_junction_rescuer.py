#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MicroDNA Junction Read Rescuer (Enhanced Multi-Modal Evidence & Soft-Clip Rescuer)
==================================================================================
"""

import argparse
import bisect
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pysam
from scipy.stats import fisher_exact, mannwhitneyu

# 自动将 conda 虚拟环境的 bin 目录加入 PATH，确保 bwa 与 samtools 始终免配置可用
_env_bin = Path(sys.executable).parent
if str(_env_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_env_bin}:{os.environ.get('PATH', '')}"

# 项目根目录自动定位
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import HG19_FA, PREDICTIONS_DIR, RAW_DATA_DIR, RESULTS_DIR
    DEFAULT_REF = HG19_FA
    DEFAULT_RAW_DIR = RAW_DATA_DIR
    DEFAULT_PRED_DIR = PREDICTIONS_DIR
    DEFAULT_OUT_DIR = RESULTS_DIR / "rescue"
except Exception:
    DEFAULT_REF = PROJECT_ROOT / "refs" / "hg19.fa"
    DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DEFAULT_PRED_DIR = PROJECT_ROOT / "results" / "predictions"
    DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "rescue"


def parse_args():
    parser = argparse.ArgumentParser(
        description="MicroDNA Junction Read Rescuer (Soft-Clip & Anchor Expansion Enhanced)"
    )
    # 输入与模式控制
    parser.add_argument("-b", "--bam", default=None,
                        help="Path to a single BAM file for single-sample mode")
    parser.add_argument("-t", "--target-bed", default=None,
                        help="Target candidate BED file (or fallback BED in batch mode)")
    parser.add_argument("--bam-dir", type=Path, default=None,
                        help="Directory containing pre-aligned BAM files (Bowtie2 or BWA) for batch mode")
    parser.add_argument("--auto", action="store_true",
                        help="Force automated FASTQ discovery and BWA alignment mode")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RAW_DIR,
                        help=f"Directory containing paired FASTQ files (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_PRED_DIR,
                        help=f"Directory containing predicted candidate BED files (default: {DEFAULT_PRED_DIR})")
    parser.add_argument("-r", "--reference", type=Path, default=DEFAULT_REF,
                        help=f"Reference genome FASTA file (default: {DEFAULT_REF})")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory for rescue reports (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("-p", "--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="CPU threads for parallel scanning (default: max - 1)")
    parser.add_argument("--force-align", action="store_true",
                        help="Force BWA re-alignment even if output BAM already exists")

    # 算法与证据判定参数
    parser.add_argument("--mode", choices=["all", "split", "everted", "softclip"], default="all",
                        help="Evidence capture mode: 'split', 'everted', 'softclip', or 'all' (default: all)")
    parser.add_argument("--search-margin", type=int, default=200,
                        help="Distance in bp to expand before and after candidate region boundaries (default: 200 bp)")
    parser.add_argument("--min-overlap", type=float, default=0.35,
                        help="Minimum reciprocal overlap ratio with circular molecule (default: 0.35)")
    parser.add_argument("--flank-tol", type=int, default=200,
                        help="Boundary tolerance in bp when validating circle endpoints (default: 200 bp)")
    parser.add_argument("--min-mapq", type=int, default=10,
                        help="Minimum MAPQ quality threshold for primary alignment (default: 10)")
    parser.add_argument("--min-clip-len", type=int, default=15,
                        help="Minimum soft-clip length in bp to evaluate for SC-junction (default: 15 bp)")
    parser.add_argument("--min-circle-len", type=int, default=150,
                        help="Minimum circular DNA length to rescue (default: 150 bp)")
    parser.add_argument("--max-circle-len", type=int, default=3000,
                        help="Maximum circular DNA length to rescue (default: 3000 bp)")
    parser.add_argument("--depth-tolerance", type=float, default=0.35,
                        help="Coverage tolerance ratio for control matching (default: 0.35)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # 置信度梯队分层 (Top-K Stratification) 增强参数
    parser.add_argument("--tiers", type=str, default="0.01,0.05,0.10,0.20,0.50,1.00",
                        help="Comma-separated Top-K stratification fractions (e.g. '0.01,0.05,0.10,0.20,0.50,1.00' or '1%%,5%%,10%%,20%%,50%%,100%%')")
    parser.add_argument("--rank-by", choices=["auto", "score", "depth"], default="auto",
                        help="Metric to rank candidate regions for stratification: 'score' (col 5/prob), 'depth' (BAM coverage), or 'auto' (default: auto)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable generation of stratification publication figures")
    return parser.parse_args()


# =====================================================================
# 1. 样本名称解析与文件关联检索模块
# =====================================================================

def ensure_bam_indexed(bam_path: Path, threads: int = 4) -> None:
    bai1 = Path(str(bam_path) + ".bai")
    bai2 = bam_path.with_suffix(".bai")
    if not (bai1.exists() or bai2.exists()):
        print(f"[*] Indexing BAM: {bam_path.name}...")
        subprocess.run(["samtools", "index", "-@", str(threads), str(bam_path)], check=True)


def get_sample_name_from_bam(bam_path: Path) -> str:
    name = bam_path.name
    if name in ["aligned.sorted.bam", "aligned.bam", "sorted.bam"]:
        return bam_path.parent.name

    suffixes = sorted([
        "_bwa.sorted.bam", ".bwa.sorted.bam",
        "_aligned.sorted.bam", ".aligned.sorted.bam",
        "_sorted.bam", ".sorted.bam",
        "_bwa.bam", ".bwa.bam",
        "_aligned.bam", ".aligned.bam",
        ".bam"
    ], key=len, reverse=True)

    clean_name = name
    for suffix in suffixes:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
            break

    clean_name = re.sub(r"([._](bwa|bowtie2|aligned|sorted))+$", "", clean_name, flags=re.IGNORECASE)
    return clean_name


def find_fastq_pairs(directory: Path) -> List[Tuple[str, Path, Path]]:
    pairs = []
    seen_samples = set()
    directory = Path(directory)

    if not directory.exists():
        return pairs

    suffix_pairs = [
        ("_1.fastq.gz", "_2.fastq.gz"),
        ("_1.fq.gz", "_2.fq.gz"),
        ("_R1.fastq.gz", "_R2.fastq.gz"),
        ("_R1.fq.gz", "_R2.fq.gz"),
        (".1.fastq.gz", ".2.fastq.gz"),
        (".1.fq.gz", ".2.fq.gz"),
        ("_1.fastq", "_2.fastq"),
        ("_1.fq", "_2.fq"),
        ("_R1.fastq", "_R2.fastq"),
        ("_R1.fq", "_R2.fq"),
        (".1.fastq", ".2.fastq"),
        (".1.fq", ".2.fq"),
    ]

    for r1_suf, r2_suf in suffix_pairs:
        for r1 in sorted(directory.glob(f"*{r1_suf}")):
            sample_name = r1.name[:-len(r1_suf)]
            if sample_name in seen_samples:
                continue
            r2 = r1.parent / f"{sample_name}{r2_suf}"
            if r2.exists():
                pairs.append((sample_name, r1, r2))
                seen_samples.add(sample_name)

    return pairs


def ensure_reference_indices(reference: Path) -> None:
    reference = Path(reference).resolve()
    if not reference.exists():
        raise FileNotFoundError(f"Reference genome FASTA not found: {reference}")

    bwa_exts = [".amb", ".ann", ".bwt", ".pac", ".sa"]
    has_bwa = all(Path(str(reference) + ext).exists() for ext in bwa_exts)
    if not has_bwa:
        if shutil.which("bwa") is None:
            raise RuntimeError("Tool 'bwa' not found in PATH. Please install bwa.")
        print(f"[*] Generating BWA index for {reference.name}...")
        subprocess.run(["bwa", "index", str(reference)], check=True)
        print("[+] BWA index built successfully.")

    fai = Path(str(reference) + ".fai")
    if not fai.exists():
        if shutil.which("samtools") is None:
            raise RuntimeError("Tool 'samtools' not found in PATH. Please install samtools.")
        print(f"[*] Generating faidx index for {reference.name}...")
        subprocess.run(["samtools", "faidx", str(reference)], check=True)
        print("[+] faidx index built successfully.")


def align_and_sort_bwa(
    r1: Path,
    r2: Path,
    reference: Path,
    output_bam: Path,
    threads: int = 8,
    force_align: bool = False,
) -> Path:
    output_bam = Path(output_bam).resolve()
    output_bam.parent.mkdir(parents=True, exist_ok=True)

    if not force_align and output_bam.exists() and output_bam.stat().st_size > 0:
        ensure_bam_indexed(output_bam, threads=threads)
        print(f"[*] Reusing existing valid BWA BAM: {output_bam.name}")
        return output_bam

    if shutil.which("bwa") is None or shutil.which("samtools") is None:
        raise RuntimeError("Missing required tool: 'bwa' or 'samtools' not found in PATH.")

    print(f"[*] Aligning {r1.name} & {r2.name} with BWA-MEM ({threads} threads)...")
    cmd = (
        f"bwa mem -t {threads} {reference} {r1} {r2} | "
        f"samtools sort -@{threads} -o {output_bam}"
    )
    res = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"BWA alignment failed:\n{res.stderr}")

    ensure_bam_indexed(output_bam, threads=threads)
    print(f"[+] BWA alignment completed -> {output_bam}")
    return output_bam


def find_target_bed_for_sample(
    sample_name: str,
    target_dir: Path,
    fallback_bed: Optional[Path] = None,
) -> Optional[Path]:
    if target_dir.exists():
        clean_name = re.sub(r"([._](bwa|bowtie2|aligned|sorted))+$", "", sample_name, flags=re.IGNORECASE)
        candidate_names = [clean_name]
        if sample_name != clean_name:
            candidate_names.append(sample_name)

        for name in candidate_names:
            candidates = [
                target_dir / f"{name}_microDNA.bed",
                target_dir / f"{name}.bed",
                target_dir / f"{name}_predictions.bed",
                target_dir / f"{name}_candidates.bed",
            ]
            for c in candidates:
                if c.exists() and c.stat().st_size > 0:
                    return c

            fuzzy = sorted(list(target_dir.glob(f"{name}*.bed")))
            for f in fuzzy:
                if f.stat().st_size > 0:
                    return f

    if fallback_bed and Path(fallback_bed).exists() and Path(fallback_bed).stat().st_size > 0:
        return Path(fallback_bed)

    return None


# =====================================================================
# 2. 染色体名称标准化与极速空间检索
# =====================================================================

def resolve_chromosome_name(query_chrom: str, target_chroms: Set[str]) -> Optional[str]:
    if query_chrom in target_chroms:
        return query_chrom
    q_lower = query_chrom.lower()
    bare = q_lower[3:] if q_lower.startswith("chr") else q_lower
    candidates = [
        query_chrom, f"chr{bare}", bare, query_chrom.upper(), query_chrom.lower(),
        f"chr{bare.upper()}", f"chr{bare.lower()}"
    ]
    for c in candidates:
        if c in target_chroms:
            return c
    return None


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for s, e in sorted_ivs[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def interval_overlaps_disjoint(
    intervals: List[Tuple[int, int]],
    q_start: int,
    q_end: int,
    starts: List[int],
) -> bool:
    if not intervals:
        return False
    idx = bisect.bisect_right(starts, q_start)
    if idx > 0:
        prev_s, prev_e = intervals[idx - 1]
        if prev_e > q_start:
            return True
    if idx < len(intervals):
        next_s, next_e = intervals[idx]
        if next_s < q_end:
            return True
    return False


def load_bed_regions(bed_path: Path, bam_chroms: Set[str]) -> List[Dict]:
    regions = []
    skipped = 0
    with open(bed_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                continue
            raw_chrom = parts[0]
            resolved_chrom = resolve_chromosome_name(raw_chrom, bam_chroms)
            if not resolved_chrom:
                skipped += 1
                continue
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            if end <= start:
                continue
            region_id = parts[3] if len(parts) > 3 else f"ROI_{idx+1}"
            
            # 尝试解析候选区置信度得分 (第 5 列，或从 ID 字符串提取 score/prob)
            score = None
            if len(parts) > 4:
                try:
                    score = float(parts[4])
                except ValueError:
                    pass
            if score is None and len(parts) > 3:
                m = re.search(r"(?:score|prob|p)[=_:]([0-9.]+)", parts[3], flags=re.IGNORECASE)
                if m:
                    try:
                        score = float(m.group(1))
                    except ValueError:
                        pass

            regions.append({
                "chrom": resolved_chrom,
                "raw_chrom": raw_chrom,
                "start": start,
                "end": end,
                "id": region_id,
                "len": end - start,
                "score": score,
            })
    if skipped > 0:
        print(f"[WARN] Skipped {skipped} regions from BED because chromosomes were absent in BAM.")
    return regions


# =====================================================================
# 3. 极速对照组采样算法
# =====================================================================

def generate_matched_controls(
    targets: List[Dict],
    bam: pysam.AlignmentFile,
    chrom_lens: Dict[str, int],
    seed: int = 42,
    depth_tol: float = 0.35,
) -> List[Dict]:
    random.seed(seed)
    raw_forbidden = defaultdict(list)
    for t in targets:
        raw_forbidden[t["chrom"]].append((max(0, t["start"] - 2000), t["end"] + 2000))

    disjoint_forbidden = {}
    starts_forbidden = {}
    for chrom, ivs in raw_forbidden.items():
        merged = merge_intervals(ivs)
        disjoint_forbidden[chrom] = merged
        starts_forbidden[chrom] = [s for s, e in merged]

    controls = []
    print(f"[*] Fast generating matched controls for {len(targets)} candidate regions...")
    read_len = 150

    for t in targets:
        chrom = t["chrom"]
        if chrom not in chrom_lens:
            continue
        c_len = chrom_lens[chrom]
        r_len = t["len"]

        try:
            t_counts = bam.count(chrom, t["start"], t["end"])
            target_depth = (t_counts * read_len) / max(1, r_len)
        except Exception:
            target_depth = 0.0

        min_depth = target_depth * (1.0 - depth_tol)
        max_depth = target_depth * (1.0 + depth_tol)
        max_start = max(10001, c_len - r_len - 10000)
        if max_start <= 10000:
            continue

        best_ctrl = None
        merged_ivs = disjoint_forbidden.get(chrom, [])
        starts_ivs = starts_forbidden.get(chrom, [])

        for _ in range(15):
            cand_start = random.randint(10000, max_start)
            cand_end = cand_start + r_len
            if interval_overlaps_disjoint(merged_ivs, cand_start, cand_end, starts_ivs):
                continue

            if target_depth > 1.0:
                try:
                    c_counts = bam.count(chrom, cand_start, cand_end)
                    c_depth = (c_counts * read_len) / max(1, r_len)
                    if not (min_depth <= c_depth <= max_depth or c_depth >= 5.0):
                        continue
                except Exception:
                    continue

            best_ctrl = {
                "chrom": chrom,
                "raw_chrom": t.get("raw_chrom", chrom),
                "start": cand_start,
                "end": cand_end,
                "id": f"CTRL_{t['id']}",
                "len": r_len,
                "score": t.get("score"),
            }
            break

        if best_ctrl is None:
            for _ in range(10):
                cand_start = random.randint(10000, max_start)
                cand_end = cand_start + r_len
                if not interval_overlaps_disjoint(merged_ivs, cand_start, cand_end, starts_ivs):
                    best_ctrl = {
                        "chrom": chrom,
                        "raw_chrom": t.get("raw_chrom", chrom),
                        "start": cand_start,
                        "end": cand_end,
                        "id": f"CTRL_{t['id']}",
                        "len": r_len,
                    }
                    break

        if best_ctrl is None:
            cand_start = random.randint(10000, max_start)
            best_ctrl = {
                "chrom": chrom,
                "raw_chrom": t.get("raw_chrom", chrom),
                "start": cand_start,
                "end": cand_end,
                "id": f"CTRL_{t['id']}",
                "len": r_len,
                "score": t.get("score"),
            }

        controls.append(best_ctrl)

    return controls


# =====================================================================
# 4. 环化拓扑与多模态证据捕获引擎 (核心改进)
# =====================================================================

def is_concordant_with_region(
    circle_start: int,
    circle_end: int,
    eff_start: int,
    eff_end: int,
    min_overlap: float = 0.35,
    flank_tol: int = 200,
) -> bool:
    """
    自适应拓扑一致性判定:
    1. 包含判定: 微环完全落在延展后的候选区间内部 (允许边界外延 flank_tol)
    2. 局部覆盖: 微环跨度在候选区上的覆盖比例达到 min_overlap
    """
    # 准则 1: 物理区间包含于宽峰内
    if circle_start >= eff_start - flank_tol and circle_end <= eff_end + flank_tol:
        return True

    # 准则 2: 覆盖微环分子自身的大部
    ov_start = max(circle_start, eff_start)
    ov_end = min(circle_end, eff_end)
    ov_len = max(0, ov_end - ov_start)
    circle_len = max(1, circle_end - circle_start)

    if ov_len / circle_len >= min_overlap:
        return True

    # 准则 3: 双端点锚定在公差窗口内
    if abs(circle_start - eff_start) <= flank_tol and abs(circle_end - eff_end) <= flank_tol:
        return True

    return False


def parse_sa_tag(sa_string: str) -> List[Dict]:
    entries = []
    for hit in sa_string.strip(";").split(";"):
        if not hit:
            continue
        fields = hit.split(",")
        if len(fields) < 5:
            continue
        try:
            entries.append({
                "chrom": fields[0],
                "pos": int(fields[1]) - 1,
                "strand": fields[2],
                "cigar": fields[3],
                "mapq": int(fields[4]),
                "nm": int(fields[5]) if len(fields) > 5 else 0
            })
        except ValueError:
            continue
    return entries


def get_sa_query_offset(cigar_str: str) -> int:
    match = re.match(r"^(\d+)[SH]", cigar_str)
    return int(match.group(1)) if match else 0


def detect_head_to_tail_split_read(
    read: pysam.AlignedSegment,
    eff_start: int,
    eff_end: int,
    bam_chroms: Set[str],
    min_overlap: float = 0.35,
    flank_tol: int = 200,
    min_mapq: int = 10,
    min_len: int = 150,
    max_len: int = 3000,
) -> Optional[Dict]:
    """捕获单碱基 Split-Read。主比对 MAPQ >= min_mapq，副比对放宽至 MAPQ >= 0"""
    if not read.has_tag("SA"):
        return None
    if read.mapping_quality < min_mapq or read.reference_start is None or read.reference_end is None:
        return None

    prim_chrom = read.reference_name
    prim_strand = "-" if read.is_reverse else "+"
    prim_start = read.reference_start
    prim_end = read.reference_end
    prim_qstart = read.query_alignment_start

    sa_hits = parse_sa_tag(read.get_tag("SA"))

    for sa in sa_hits:
        sa_chrom = resolve_chromosome_name(sa["chrom"], bam_chroms)
        if sa_chrom != prim_chrom or sa["strand"] != prim_strand:
            continue

        sa_start = sa["pos"]
        sa_ref_span = sum(int(length) for length, op in re.findall(r"(\d+)([MDN=X])", sa["cigar"]))
        sa_end = sa_start + sa_ref_span
        sa_qstart = get_sa_query_offset(sa["cigar"])

        junc_left = min(prim_start, sa_start)
        junc_right = max(prim_end, sa_end)
        circle_len = junc_right - junc_left

        if not (min_len <= circle_len <= max_len):
            continue

        if not is_concordant_with_region(junc_left, junc_right, eff_start, eff_end, min_overlap, flank_tol):
            continue

        is_h2t = False
        if prim_strand == "+":
            if prim_qstart < sa_qstart and prim_start > sa_start:
                is_h2t = True
            elif sa_qstart < prim_qstart and sa_start > prim_start:
                is_h2t = True
        else:
            if prim_qstart < sa_qstart and prim_start < sa_start:
                is_h2t = True
            elif sa_qstart < prim_qstart and sa_start < prim_start:
                is_h2t = True

        if is_h2t:
            return {
                "qname": read.query_name,
                "evidence_type": "Split-Read",
                "chrom": prim_chrom,
                "junction_start": junc_left,
                "junction_end": junc_right,
                "circle_len": circle_len,
                "prim_coords": f"{prim_start}-{prim_end}",
                "split_coords": f"{sa_start}-{sa_end}",
                "mapq": f"{read.mapping_quality},{sa['mapq']}",
            }

    return None


def detect_everted_mate_pair(
    read: pysam.AlignedSegment,
    eff_start: int,
    eff_end: int,
    min_overlap: float = 0.35,
    flank_tol: int = 200,
    min_mapq: int = 10,
    min_len: int = 150,
    max_len: int = 3000,
) -> Optional[Dict]:
    """捕获跨接头外翻配对 (Everted Pair, RF 拓扑)"""
    if not read.is_paired or read.is_unmapped or read.mate_is_unmapped:
        return None
    if read.is_proper_pair:
        return None
    if read.mapping_quality < min_mapq:
        return None
    if read.reference_id != read.next_reference_id:
        return None

    chrom = read.reference_name
    r_start = read.reference_start
    r_end = read.reference_end
    m_start = read.next_reference_start

    # 正链 (+) 在下游，负链 mate (-) 在上游
    if not read.is_reverse and read.mate_is_reverse:
        if r_start > m_start:
            span = (r_end if r_end else r_start + 150) - m_start
            if not (min_len <= span <= max_len):
                return None
            circle_s = m_start
            circle_e = r_end if r_end else r_start + 150
            if is_concordant_with_region(circle_s, circle_e, eff_start, eff_end, min_overlap, flank_tol):
                return {
                    "qname": read.query_name,
                    "evidence_type": "Everted-Pair",
                    "chrom": chrom,
                    "junction_start": circle_s,
                    "junction_end": circle_e,
                    "circle_len": span,
                    "prim_coords": f"{r_start}-{r_end}",
                    "split_coords": f"mate:{m_start}",
                    "mapq": f"{read.mapping_quality}",
                }

    # 负链 (-) 在上游，正链 mate (+) 在下游
    elif read.is_reverse and not read.mate_is_reverse:
        if m_start > r_start:
            span = (m_start + 150) - r_start
            if not (min_len <= span <= max_len):
                return None
            circle_s = r_start
            circle_e = m_start + 150
            if is_concordant_with_region(circle_s, circle_e, eff_start, eff_end, min_overlap, flank_tol):
                return {
                    "qname": read.query_name,
                    "evidence_type": "Everted-Pair",
                    "chrom": chrom,
                    "junction_start": circle_s,
                    "junction_end": circle_e,
                    "circle_len": span,
                    "prim_coords": f"{r_start}-{r_end}",
                    "split_coords": f"mate:{m_start}",
                    "mapq": f"{read.mapping_quality}",
                }

    return None


def detect_softclip_realigned_junction(
    read: pysam.AlignedSegment,
    ref_fasta: pysam.FastaFile,
    chrom: str,
    eff_start: int,
    eff_end: int,
    min_clip_len: int = 15,
    min_mapq: int = 10,
    min_len: int = 150,
    max_len: int = 3000,
) -> Optional[Dict]:
    """
    【新增核心引擎】Soft-clip 剪切接头重比对验证 (SC-Junction)。
    捕获单侧软剪切 >= 15 bp 的读段，验证其剪切序列是否与对侧边界发生闭合环接。
    """
    if read.is_unmapped or read.mapping_quality < min_mapq:
        return None
    if not read.cigartuples or not read.query_sequence:
        return None

    r_start = read.reference_start
    r_end = read.reference_end
    if r_start is None or r_end is None:
        return None

    cigar = read.cigartuples
    left_clip = cigar[0][1] if cigar[0][0] == 4 else 0
    right_clip = cigar[-1][1] if cigar[-1][0] == 4 else 0

    if left_clip < min_clip_len and right_clip < min_clip_len:
        return None

    qseq = read.query_sequence
    ref_len = ref_fasta.get_reference_length(chrom)

    # 情况 A: 右端软切除 (读段测到微环右断裂点，延伸碱基进入左断裂点)
    if right_clip >= min_clip_len:
        clip_seq = qseq[-right_clip:]
        probe_len = min(len(clip_seq), 32)
        probe = clip_seq[:probe_len]

        # 在候选区左边界窗口探测
        search_s = max(0, eff_start - 100)
        search_e = min(ref_len, eff_start + 300)
        try:
            target_dna = ref_fasta.fetch(chrom, search_s, search_e).upper()
            idx = target_dna.find(probe)
            if idx != -1:
                junc_left = search_s + idx
                junc_right = r_end
                circle_len = junc_right - junc_left
                if min_len <= circle_len <= max_len:
                    return {
                        "qname": read.query_name,
                        "evidence_type": "SoftClip-Junction",
                        "chrom": chrom,
                        "junction_start": junc_left,
                        "junction_end": junc_right,
                        "circle_len": circle_len,
                        "prim_coords": f"{r_start}-{r_end}",
                        "split_coords": f"clip_match:{junc_left}",
                        "mapq": f"{read.mapping_quality}",
                    }
        except Exception:
            pass

    # 情况 B: 左端软切除 (读段测到微环左断裂点，前端碱基来自右断裂点)
    if left_clip >= min_clip_len:
        clip_seq = qseq[:left_clip]
        probe_len = min(len(clip_seq), 32)
        probe = clip_seq[-probe_len:]

        # 在候选区右边界窗口探测
        search_s = max(0, eff_end - 300)
        search_e = min(ref_len, eff_end + 100)
        try:
            target_dna = ref_fasta.fetch(chrom, search_s, search_e).upper()
            idx = target_dna.find(probe)
            if idx != -1:
                junc_left = r_start
                junc_right = search_s + idx + probe_len
                circle_len = junc_right - junc_left
                if min_len <= circle_len <= max_len:
                    return {
                        "qname": read.query_name,
                        "evidence_type": "SoftClip-Junction",
                        "chrom": chrom,
                        "junction_start": junc_left,
                        "junction_end": junc_right,
                        "circle_len": circle_len,
                        "prim_coords": f"{r_start}-{r_end}",
                        "split_coords": f"clip_match:{junc_right}",
                        "mapq": f"{read.mapping_quality}",
                    }
        except Exception:
            pass

    return None


def calculate_microhomology(
    ref_fasta: pysam.FastaFile,
    chrom: str,
    junc_start: int,
    junc_end: int,
    max_search: int = 25,
) -> int:
    try:
        c_len = ref_fasta.get_reference_length(chrom)
        s_left = max(0, junc_start - max_search)
        e_left = min(c_len, junc_start + max_search)
        s_right = max(0, junc_end - max_search)
        e_right = min(c_len, junc_end + max_search)

        seq_left = ref_fasta.fetch(chrom, s_left, e_left).upper()
        seq_right = ref_fasta.fetch(chrom, s_right, e_right).upper()

        idx_left = junc_start - s_left
        idx_right = junc_end - s_right

        left_match = 0
        for k in range(1, max_search + 1):
            if idx_left - k >= 0 and idx_right - k >= 0:
                if seq_left[idx_left - k] == seq_right[idx_right - k]:
                    left_match += 1
                else:
                    break
            else:
                break

        right_match = 0
        for k in range(max_search):
            if idx_left + k < len(seq_left) and idx_right + k < len(seq_right):
                if seq_left[idx_left + k] == seq_right[idx_right + k]:
                    right_match += 1
                else:
                    break
            else:
                break

        return max(left_match, right_match)
    except Exception:
        return 0


# =====================================================================
# 5. 高性能多进程流式扫描调度器
# =====================================================================

def _scan_chunk_worker(
    chunk_regions: List[Dict],
    bam_path: str,
    ref_path: str,
    group_type: str,
    mode: str,
    search_margin: int,
    min_overlap: float,
    flank_tol: int,
    min_mapq: int,
    min_clip_len: int,
    min_circle_len: int,
    max_circle_len: int,
    bam_chroms: Set[str],
) -> Tuple[List[Dict], List[Dict], int]:
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref_fasta = pysam.FastaFile(ref_path)

    region_records = []
    rescued_reads = []
    sa_tags_seen = 0

    for reg in chunk_regions:
        chrom = reg["chrom"]
        start = reg["start"]
        end = reg["end"]
        reg_len = reg["len"]

        # 前后增加搜索裕量，全面扩大覆盖
        eff_start = max(0, start - search_margin)
        eff_end = end + search_margin

        search_start = max(0, eff_start - 250)
        search_end = eff_end + 250

        rescued_in_this_region = {}
        total_overlap_bases = 0

        try:
            for read in bam.fetch(chrom, search_start, search_end):
                if read.is_unmapped or read.is_duplicate:
                    continue

                r_s = read.reference_start
                r_e = read.reference_end
                if r_s is not None and r_e is not None and r_s < end and r_e > start:
                    total_overlap_bases += min(r_e, end) - max(r_s, start)

                if read.has_tag("SA"):
                    sa_tags_seen += 1

                junc_info = None

                # 1. 优先捕获 Split Read
                if mode in ["all", "split"]:
                    junc_info = detect_head_to_tail_split_read(
                        read, eff_start, eff_end, bam_chroms,
                        min_overlap=min_overlap, flank_tol=flank_tol,
                        min_mapq=min_mapq, min_len=min_circle_len,
                        max_len=max_circle_len
                    )

                # 2. 尝试捕获 Soft-Clip 重比对闭合证据
                if not junc_info and mode in ["all", "softclip"]:
                    junc_info = detect_softclip_realigned_junction(
                        read, ref_fasta, chrom, eff_start, eff_end,
                        min_clip_len=min_clip_len, min_mapq=min_mapq,
                        min_len=min_circle_len, max_len=max_circle_len
                    )

                # 3. 备选捕获 Everted Pair
                if not junc_info and mode in ["all", "everted"]:
                    junc_info = detect_everted_mate_pair(
                        read, eff_start, eff_end,
                        min_overlap=min_overlap, flank_tol=flank_tol,
                        min_mapq=min_mapq, min_len=min_circle_len,
                        max_len=max_circle_len
                    )

                if junc_info:
                    qname = junc_info["qname"]
                    if qname not in rescued_in_this_region:
                        m_hom = 0
                        if junc_info["evidence_type"] == "Split-Read":
                            m_hom = calculate_microhomology(
                                ref_fasta, chrom, junc_info["junction_start"], junc_info["junction_end"]
                            )
                        junc_info["microhomology_bp"] = m_hom
                        junc_info["region_id"] = reg["id"]
                        junc_info["group"] = group_type
                        rescued_in_this_region[qname] = junc_info

        except Exception:
            pass

        count = len(rescued_in_this_region)
        mean_depth = total_overlap_bases / max(1, reg_len)

        region_records.append({
            "region_id": reg["id"],
            "group": group_type,
            "chrom": reg.get("raw_chrom", chrom),
            "start": start,
            "end": end,
            "length": reg_len,
            "mean_depth": round(mean_depth, 2),
            "rescued_reads": count,
            "has_junction": 1 if count > 0 else 0,
            "score": reg.get("score"),
        })
        rescued_reads.extend(rescued_in_this_region.values())

    bam.close()
    ref_fasta.close()
    return region_records, rescued_reads, sa_tags_seen


def scan_cohort_regions_parallel(
    regions: List[Dict],
    bam_path: Path,
    ref_path: Path,
    group_type: str,
    args,
    bam_chroms: Set[str],
) -> Tuple[List[Dict], List[Dict], int]:
    if not regions:
        return [], [], 0

    workers = min(args.threads, len(regions))
    if workers <= 1:
        return _scan_chunk_worker(
            regions, str(bam_path), str(ref_path), group_type, args.mode,
            args.search_margin, args.min_overlap, args.flank_tol, args.min_mapq,
            args.min_clip_len, args.min_circle_len, args.max_circle_len, bam_chroms
        )

    chunk_size = int(np.ceil(len(regions) / workers))
    chunks = [regions[i:i + chunk_size] for i in range(0, len(regions), chunk_size)]

    all_region_records = []
    all_rescued_reads = []
    total_sa = 0

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _scan_chunk_worker,
                c, str(bam_path), str(ref_path), group_type, args.mode,
                args.search_margin, args.min_overlap, args.flank_tol, args.min_mapq,
                args.min_clip_len, args.min_circle_len, args.max_circle_len, bam_chroms
            )
            for c in chunks
        ]
        for f in as_completed(futures):
            recs, reads, sa = f.result()
            all_region_records.extend(recs)
            all_rescued_reads.extend(reads)
            total_sa += sa

    id_order = {r["id"]: i for i, r in enumerate(regions)}
    all_region_records.sort(key=lambda x: id_order.get(x["region_id"], 0))

    return all_region_records, all_rescued_reads, total_sa


# =====================================================================
# 6. 置信度梯队分层 (Top-K Stratification) 与剂量反应分析引擎
# =====================================================================

def parse_tier_fractions(tiers_str: str) -> List[float]:
    """解析 0.01,0.05,0.10,0.20,0.50,1.00 或 1%,5%,10%,20%,50%,100% 格式分层"""
    fractions = []
    for raw in str(tiers_str).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw.endswith("%"):
            val = float(raw[:-1]) / 100.0
        else:
            val = float(raw)
            if val > 1.0:
                val = val / 100.0
        val = max(0.0001, min(1.0, val))
        fractions.append(val)
    return sorted(list(set(fractions)))


def compute_top_k_stratification(
    target_summary: List[Dict],
    control_summary: List[Dict],
    target_junctions: List[Dict],
    control_junctions: List[Dict],
    tiers: List[float],
    rank_by: str = "auto",
) -> Tuple[List[Dict], str]:
    """
    根据置信度得分 (score) 或测序深度 (mean_depth) 将候选微环分层为 Top K 梯队，
    计算各梯队在非富集 WGS 背景下的接头直接证据捕获率、富集倍数及统计显著性。
    """
    if not target_summary:
        return [], "none"

    valid_scores = [r.get("score") for r in target_summary if r.get("score") is not None]
    if rank_by == "score" or (rank_by == "auto" and len(valid_scores) >= len(target_summary) * 0.3):
        rank_key = "score"
        metric_name = "Confidence Score (Prob)"
    else:
        rank_key = "mean_depth"
        metric_name = "Sequencing Read Depth (mean_depth)"

    # 配对靶向区域与严格匹配的背景对照区域
    paired = list(zip(target_summary, control_summary))
    paired.sort(
        key=lambda p: (p[0].get(rank_key) if p[0].get(rank_key) is not None else -999999.0),
        reverse=True
    )

    # 建立 region_id -> 证据类型计数字典
    target_ev_map = defaultdict(lambda: {"Split-Read": 0, "SoftClip-Junction": 0, "Everted-Pair": 0})
    for j in target_junctions:
        ev_t = j.get("evidence_type", "Other")
        target_ev_map[j["region_id"]][ev_t] += 1

    total_n = len(paired)
    bg_c_pos = sum(r["has_junction"] for r in control_summary)
    bg_c_rate = (bg_c_pos / max(1, len(control_summary))) * 100.0

    strat_records = []

    for frac in tiers:
        k = max(1, int(np.ceil(total_n * frac)))
        tier_pairs = paired[:k]
        tier_targets = [p[0] for p in tier_pairs]
        tier_controls = [p[1] for p in tier_pairs]

        t_pos = sum(r["has_junction"] for r in tier_targets)
        t_rate = (t_pos / k) * 100.0

        c_pos = sum(r["has_junction"] for r in tier_controls)
        c_rate = (c_pos / k) * 100.0

        enrichment_fold_matched = round(t_rate / max(c_rate, 0.05), 2)
        enrichment_fold_bg = round(t_rate / max(bg_c_rate, 0.05), 2)

        table = [[t_pos, k - t_pos], [c_pos, k - c_pos]]
        try:
            odds_ratio, p_val = fisher_exact(table, alternative="greater")
        except Exception:
            odds_ratio, p_val = 1.0, 1.0

        split_cnt = sum(target_ev_map[r["region_id"]]["Split-Read"] for r in tier_targets)
        clip_cnt = sum(target_ev_map[r["region_id"]]["SoftClip-Junction"] for r in tier_targets)
        evert_cnt = sum(target_ev_map[r["region_id"]]["Everted-Pair"] for r in tier_targets)

        cutoff_val = tier_targets[-1].get(rank_key)
        cutoff_str = f"{cutoff_val:.4f}" if cutoff_val is not None else "N/A"

        pct_label = f"Top {int(round(frac * 100))}%" if frac < 1.0 else "Top 100% (All)"
        strat_records.append({
            "tier": pct_label,
            "tier_name": pct_label,
            "fraction": frac,
            "tier_fraction": frac,
            "rank_metric": metric_name,
            "rank_by": rank_key,
            "cutoff_threshold": cutoff_str,
            "target_count": k,
            "k_count": k,
            "target_rescued": t_pos,
            "target_rate_pct": round(t_rate, 2),
            "target_rescue_rate_pct": round(t_rate, 2),
            "control_count": k,
            "control_rescued": c_pos,
            "control_rate_pct": round(c_rate, 2),
            "control_rescue_rate_pct": round(c_rate, 2),
            "enrichment_fold": enrichment_fold_matched,
            "enrichment_fold_matched": enrichment_fold_matched,
            "bg_control_rate_pct": round(bg_c_rate, 2),
            "enrichment_fold_bg": enrichment_fold_bg,
            "odds_ratio": round(odds_ratio, 2),
            "fisher_pval": p_val,
            "fisher_pvalue": p_val,
            "split_reads": split_cnt,
            "softclip_reads": clip_cnt,
            "everted_pairs": evert_cnt,
        })

    return strat_records, metric_name


def plot_stratification_chart(strat_records: List[Dict], out_prefix: Path, title: str = "Candidate Stratification"):
    """绘制符合学术出版规范的 Top-K 置信度梯队验证图表 (双面板: 验证率对比与富集倍数阶梯)"""
    if not strat_records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["tier"] for r in strat_records]
    t_rates = [r["target_rescue_rate_pct"] for r in strat_records]
    c_rates = [r["control_rescue_rate_pct"] for r in strat_records]
    folds = [r["enrichment_fold_matched"] for r in strat_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # 1. 验证率柱状对比图 (Panel A)
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, t_rates, width, label="Target (Prioritized)", color="#D9381E", alpha=0.9, edgecolor="black", linewidth=0.8)
    bars2 = ax1.bar(x + width/2, c_rates, width, label="Matched Control", color="#4682B4", alpha=0.9, edgecolor="black", linewidth=0.8)

    ax1.set_xlabel("Confidence Stratification Tiers", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Direct Junction Rescue Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Circular Junction Verification Rate", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for b in bars1:
        h = b.get_height()
        if h > 0:
            ax1.text(b.get_x() + b.get_width()/2., h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # 2. 富集倍数阶梯折线图 (Panel B)
    ax2.plot(x, folds, marker="o", color="#2E8B57", linewidth=2.5, markersize=8, label="Enrichment Fold (Target / Control)")
    ax2.set_xlabel("Confidence Stratification Tiers", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Evidence Enrichment Fold (×)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Evidence Enrichment Gradient", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=10)

    for i, f_val in enumerate(folds):
        ax2.annotate(f"{f_val:.1f}×", (x[i], folds[i]), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9, fontweight="bold", color="#1B4D3E")

    plt.suptitle(f"{title} - MicroDNA Direct Junction Verification", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png = Path(f"{out_prefix}_stratification_plot.png")
    out_pdf = Path(f"{out_prefix}_stratification_plot.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 7. 单样本接头挽救评测工作流
# =====================================================================

def process_single_sample(
    sample_name: str,
    bam_file: Path,
    target_bed_file: Path,
    reference_file: Path,
    output_dir: Path,
    args,
) -> Optional[Dict]:
    print(f"\n{'='*75}\n[*] Processing Sample: {sample_name}\n{'='*75}")
    print(f"  - BAM File      : {bam_file}")
    print(f"  - Target BED    : {target_bed_file}")
    print(f"  - Search Margin : ±{args.search_margin} bp")

    ensure_bam_indexed(bam_file, threads=args.threads)

    bam = pysam.AlignmentFile(str(bam_file), "rb")
    ref_fasta = pysam.FastaFile(str(reference_file))

    bam_chroms = set(bam.references)
    chrom_lens = dict(zip(ref_fasta.references, ref_fasta.lengths))

    targets = load_bed_regions(target_bed_file, bam_chroms)
    if not targets:
        print(f"[ERROR] No valid regions found in {target_bed_file} matching BAM contigs. Skipping.")
        bam.close()
        ref_fasta.close()
        return None

    controls = generate_matched_controls(
        targets, bam, chrom_lens, seed=args.seed, depth_tol=args.depth_tolerance
    )

    sample_out_prefix = output_dir / sample_name
    ctrl_bed_path = f"{sample_out_prefix}_matched_controls.bed"
    with open(ctrl_bed_path, "w", encoding="utf-8") as f:
        for c in controls:
            f.write(f"{c['raw_chrom']}\t{c['start']}\t{c['end']}\t{c['id']}\n")

    bam.close()
    ref_fasta.close()

    print(f"[*] Scanning Target ROIs (n={len(targets)}) [Threads: {args.threads}, Margin: ±{args.search_margin}bp]...")
    target_summary, target_junctions, target_sa_count = scan_cohort_regions_parallel(
        targets, bam_file, reference_file, "Target", args, bam_chroms
    )

    print(f"[*] Scanning Control ROIs (n={len(controls)}) [Threads: {args.threads}, Margin: ±{args.search_margin}bp]...")
    control_summary, control_junctions, control_sa_count = scan_cohort_regions_parallel(
        controls, bam_file, reference_file, "Control", args, bam_chroms
    )

    all_summary = target_summary + control_summary
    all_junctions = target_junctions + control_junctions

    summary_tsv = f"{sample_out_prefix}_region_summary.tsv"
    with open(summary_tsv, "w", encoding="utf-8") as f:
        f.write("region_id\tgroup\tchrom\tstart\tend\tlength\tscore\tmean_depth\trescued_reads\thas_junction\n")
        for r in all_summary:
            score_val = r.get("score")
            score_str = f"{float(score_val):.4f}" if score_val is not None else "0.0000"
            f.write(f"{r['region_id']}\t{r['group']}\t{r['chrom']}\t{r['start']}\t{r['end']}\t"
                    f"{r['length']}\t{score_str}\t{r['mean_depth']}\t{r['rescued_reads']}\t{r['has_junction']}\n")

    junctions_tsv = f"{sample_out_prefix}_rescued_junctions.tsv"
    with open(junctions_tsv, "w", encoding="utf-8") as f:
        f.write("qname\tevidence_type\tgroup\tregion_id\tchrom\tjunction_start\tjunction_end\t"
                "circle_len\tprim_coords\tsplit_coords\tmapq\tmicrohomology_bp\n")
        for j in all_junctions:
            f.write(f"{j['qname']}\t{j['evidence_type']}\t{j['group']}\t{j['region_id']}\t{j['chrom']}\t"
                    f"{j['junction_start']}\t{j['junction_end']}\t{j['circle_len']}\t"
                    f"{j['prim_coords']}\t{j['split_coords']}\t{j['mapq']}\t{j['microhomology_bp']}\n")

    # 计算 Top-K 分层评测
    tier_fractions = parse_tier_fractions(getattr(args, "tiers", "0.01,0.05,0.10,0.20,0.50,1.00"))
    stratification_results, metric_used = compute_top_k_stratification(
        target_summary=target_summary,
        control_summary=control_summary,
        target_junctions=target_junctions,
        control_junctions=control_junctions,
        tiers=tier_fractions,
        rank_by=getattr(args, "rank_by", "auto"),
    )

    strat_tsv = f"{sample_out_prefix}_stratification_summary.tsv"
    strat_csv = f"{sample_out_prefix}_stratification_summary.csv"
    strat_headers = [
        "tier_name", "tier_fraction", "rank_by", "k_count",
        "target_rescued", "target_rate_pct", "control_rescued", "control_rate_pct",
        "enrichment_fold", "odds_ratio", "fisher_pval"
    ]
    with open(strat_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(strat_headers) + "\n")
        for s in stratification_results:
            f.write("\t".join(str(s[h]) for h in strat_headers) + "\n")

    with open(strat_csv, "w", encoding="utf-8") as f:
        f.write(",".join(strat_headers) + "\n")
        for s in stratification_results:
            f.write(",".join(str(s[h]) for h in strat_headers) + "\n")

    if not getattr(args, "no_plot", False) and stratification_results:
        plot_stratification_chart(
            stratification_results,
            sample_out_prefix,
            title=f"MicroDNA Junction Rescue Stratification ({sample_name})"
        )

    t_pos = sum(r["has_junction"] for r in target_summary)
    t_neg = len(target_summary) - t_pos
    c_pos = sum(r["has_junction"] for r in control_summary)
    c_neg = len(control_summary) - c_pos

    table = [[t_pos, t_neg], [c_pos, c_neg]]
    try:
        odds_ratio, p_value = fisher_exact(table, alternative="greater")
    except Exception:
        odds_ratio, p_value = 1.0, 1.0

    t_reads = [r["rescued_reads"] for r in target_summary]
    c_reads = [r["rescued_reads"] for r in control_summary]

    try:
        if all(x == 0 for x in t_reads) and all(x == 0 for x in c_reads):
            u_stat, u_pval = 0.0, 1.0
        else:
            u_stat, u_pval = mannwhitneyu(t_reads, c_reads, alternative="greater")
    except Exception:
        u_stat, u_pval = 0.0, 1.0

    t_rate = (t_pos / len(target_summary)) * 100 if target_summary else 0.0
    c_rate = (c_pos / len(control_summary)) * 100 if control_summary else 0.0

    t_hom = [j["microhomology_bp"] for j in target_junctions if j["evidence_type"] == "Split-Read"]
    avg_hom = float(np.mean(t_hom)) if t_hom else 0.0

    split_cnt = sum(1 for j in target_junctions if j["evidence_type"] == "Split-Read")
    clip_cnt = sum(1 for j in target_junctions if j["evidence_type"] == "SoftClip-Junction")
    evert_cnt = sum(1 for j in target_junctions if j["evidence_type"] == "Everted-Pair")

    print("\n" + "-" * 70)
    print(f"[{sample_name}] ENRICHMENT EVALUATION REPORT")
    print("-" * 70)
    print(f"{'Cohort':<12}{'Total ROIs':<15}{'With Evidence (>=1)':<22}{'Rescue Rate (%)':<15}")
    print(f"{'Target':<12}{len(target_summary):<15}{t_pos:<22}{t_rate:<15.2f}")
    print(f"{'Control':<12}{len(control_summary):<15}{c_pos:<22}{c_rate:<15.2f}")
    print(f"[*] Target Evidence: {split_cnt} Split-Reads, {clip_cnt} SoftClip-Junctions, {evert_cnt} Everted-Pairs")
    print(f"[*] Fisher's Exact Test Odds Ratio (OR): {odds_ratio:.3f}")
    print(f"[*] Fisher's Exact Test p-value:         {p_value:.3e}")
    print(f"[*] Mann-Whitney U Test p-value:         {u_pval:.3e}")
    if split_cnt > 0:
        print(f"[*] Average Microhomology (Split-reads): {avg_hom:.2f} bp")
    print("-" * 70)

    print("\n" + "-" * 88)
    print(f"[{sample_name}] TOP-K CONFIDENCE STRATIFICATION REPORT")
    print("-" * 88)
    print(f"{'Tier':<12}{'Rank By':<10}{'Top-K':<8}{'Target%':<12}{'Ctrl%':<10}{'Enrichment':<12}{'OR':<10}{'Fisher p':<11}")
    print("-" * 88)
    for s in stratification_results:
        print(f"{s['tier_name']:<12}{s['rank_by']:<10}{s['k_count']:<8}{s['target_rate_pct']:<12.2f}"
              f"{s['control_rate_pct']:<10.2f}{s['enrichment_fold']:<12.2f}{s['odds_ratio']:<10.2f}{s['fisher_pval']:<11.2e}")
    print("-" * 88 + "\n")

    return {
        "sample": sample_name,
        "total_targets": len(target_summary),
        "target_rescued": t_pos,
        "target_rate_pct": round(t_rate, 2),
        "control_rescued": c_pos,
        "control_rate_pct": round(c_rate, 2),
        "split_reads": split_cnt,
        "softclip_reads": clip_cnt,
        "everted_pairs": evert_cnt,
        "odds_ratio": round(odds_ratio, 3),
        "fisher_pval": p_value,
        "mwu_pval": u_pval,
        "avg_microhomology": round(avg_hom, 2),
        "stratification": stratification_results,
    }


def export_batch_cohort_summary(summary_records: List[Dict], out_dir: Path, args) -> None:
    if not summary_records:
        return

    # 1. 基础批处理汇总 (不包含嵌套的 stratification 列表)
    batch_summary_tsv = out_dir / "batch_rescue_summary.tsv"
    scalar_headers = [k for k in summary_records[0].keys() if k != "stratification"]
    with open(batch_summary_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(scalar_headers) + "\n")
        for r in summary_records:
            f.write("\t".join(str(r[h]) for h in scalar_headers) + "\n")

    print("\n" + "=" * 95)
    print("                 BATCH RESCUE EVALUATION MASTER SUMMARY")
    print("=" * 95)
    print(f"{'Sample':<16}{'Targets':<9}{'Target%':<9}{'Ctrl%':<8}{'Split':<7}{'SC-Clip':<9}{'Everted':<9}{'OR':<7}{'Fisher p':<11}")
    print("-" * 95)
    for r in summary_records:
        print(f"{r['sample']:<16}{r['total_targets']:<9}{r['target_rate_pct']:<9.2f}{r['control_rate_pct']:<8.2f}"
              f"{r['split_reads']:<7}{r['softclip_reads']:<9}{r['everted_pairs']:<9}{r['odds_ratio']:<7.2f}{r['fisher_pval']:<11.2e}")
    print("=" * 95)
    print(f"[+] Master summary report saved to: {batch_summary_tsv}\n")

    # 2. 检查是否有分层数据
    samples_with_strat = [r for r in summary_records if "stratification" in r and r["stratification"]]
    if not samples_with_strat:
        return

    sample_strat_lists = [r["stratification"] for r in samples_with_strat]
    num_tiers = len(sample_strat_lists[0])
    cohort_strat = []

    for idx in range(num_tiers):
        tier_name = sample_strat_lists[0][idx]["tier_name"]
        tier_frac = sample_strat_lists[0][idx]["tier_fraction"]
        rank_by = sample_strat_lists[0][idx]["rank_by"]

        t_rates = [s[idx]["target_rate_pct"] for s in sample_strat_lists]
        c_rates = [s[idx]["control_rate_pct"] for s in sample_strat_lists]
        enr_folds = [s[idx]["enrichment_fold"] for s in sample_strat_lists]
        ors = [s[idx]["odds_ratio"] for s in sample_strat_lists]
        k_counts = [s[idx]["k_count"] for s in sample_strat_lists]

        mean_t_rate = float(np.mean(t_rates))
        std_t_rate = float(np.std(t_rates)) if len(t_rates) > 1 else 0.0

        mean_c_rate = float(np.mean(c_rates))
        std_c_rate = float(np.std(c_rates)) if len(c_rates) > 1 else 0.0

        mean_enr = float(np.mean(enr_folds))
        std_enr = float(np.std(enr_folds)) if len(enr_folds) > 1 else 0.0

        median_or = float(np.median(ors))
        avg_k = int(round(np.mean(k_counts)))

        cohort_strat.append({
            "tier": tier_name,
            "tier_name": tier_name,
            "fraction": tier_frac,
            "tier_fraction": tier_frac,
            "rank_by": rank_by,
            "mean_k_count": avg_k,
            "k_count": avg_k,
            "mean_target_rate_pct": round(mean_t_rate, 2),
            "target_rate_pct": round(mean_t_rate, 2),
            "target_rescue_rate_pct": round(mean_t_rate, 2),
            "std_target_rate_pct": round(std_t_rate, 2),
            "mean_control_rate_pct": round(mean_c_rate, 2),
            "control_rate_pct": round(mean_c_rate, 2),
            "control_rescue_rate_pct": round(mean_c_rate, 2),
            "std_control_rate_pct": round(std_c_rate, 2),
            "mean_enrichment_fold": round(mean_enr, 2),
            "enrichment_fold": round(mean_enr, 2),
            "enrichment_fold_matched": round(mean_enr, 2),
            "std_enrichment_fold": round(std_enr, 2),
            "median_odds_ratio": round(median_or, 2),
            "odds_ratio": round(median_or, 2),
            "fisher_pval": 0.0,
            "fisher_pvalue": 0.0,
        })

    cohort_tsv = out_dir / "cohort_stratification_summary.tsv"
    cohort_csv = out_dir / "cohort_stratification_summary.csv"
    cohort_headers = [
        "tier_name", "tier_fraction", "rank_by", "mean_k_count",
        "mean_target_rate_pct", "std_target_rate_pct",
        "mean_control_rate_pct", "std_control_rate_pct",
        "mean_enrichment_fold", "std_enrichment_fold", "median_odds_ratio"
    ]

    with open(cohort_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(cohort_headers) + "\n")
        for s in cohort_strat:
            f.write("\t".join(str(s[h]) for h in cohort_headers) + "\n")

    with open(cohort_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cohort_headers) + "\n")
        for s in cohort_strat:
            f.write(",".join(str(s[h]) for h in cohort_headers) + "\n")

    print("\n" + "=" * 105)
    print("                     COHORT TOP-K STRATIFICATION SUMMARY (Mean ± Std)")
    print("=" * 105)
    print(f"{'Tier':<12}{'Rank By':<10}{'Mean-K':<8}{'Target Rate (%)':<24}{'Control Rate (%)':<22}{'Enrichment Fold':<20}{'Median OR':<10}")
    print("-" * 105)
    for s in cohort_strat:
        t_str = f"{s['mean_target_rate_pct']:.2f} ± {s['std_target_rate_pct']:.2f}"
        c_str = f"{s['mean_control_rate_pct']:.2f} ± {s['std_control_rate_pct']:.2f}"
        e_str = f"{s['mean_enrichment_fold']:.2f} ± {s['std_enrichment_fold']:.2f}"
        print(f"{s['tier_name']:<12}{s['rank_by']:<10}{s['mean_k_count']:<8}{t_str:<24}{c_str:<22}{e_str:<20}{s['median_odds_ratio']:<10.2f}")
    print("=" * 105)
    print(f"[+] Cohort stratification summary saved to: {cohort_tsv}")
    print(f"[+] Cohort stratification CSV saved to:     {cohort_csv}\n")

    if not getattr(args, "no_plot", False) and cohort_strat:
        cohort_prefix = out_dir / "cohort_overall"
        plot_stratification_chart(
            cohort_strat,
            cohort_prefix,
            title="Cohort Overall: MicroDNA Junction Rescue Stratification"
        )


# =====================================================================
# 7. 全流程自动化总控 (Main Entry)
# =====================================================================

def main():
    args = parse_args()
    ref_path = Path(args.reference).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dir = Path(args.target_dir).resolve()

    ensure_reference_indices(ref_path)

    # 模式 A: 单 BAM 显式指定模式 (-b sample.bam)
    if args.bam is not None:
        bam_file = Path(args.bam).resolve()
        if not bam_file.exists():
            print(f"[ERROR] Specified BAM file does not exist: {bam_file}", file=sys.stderr)
            sys.exit(1)

        sample_name = get_sample_name_from_bam(bam_file)
        target_bed = find_target_bed_for_sample(
            sample_name, target_dir, fallback_bed=Path(args.target_bed) if args.target_bed else None
        )
        if not target_bed:
            print(f"[ERROR] Cannot find target BED for {sample_name}. Please pass '-t <bed>'.", file=sys.stderr)
            sys.exit(1)

        process_single_sample(sample_name, bam_file, target_bed, ref_path, out_dir, args)
        return

    # 模式 B: BAM 文件夹批量处理模式 (--bam-dir <dir>)
    if args.bam_dir is not None and not args.auto:
        bams_dir = Path(args.bam_dir).resolve()
        if not bams_dir.exists():
            print(f"[ERROR] Specified BAM directory does not exist: {bams_dir}", file=sys.stderr)
            sys.exit(1)

        bams = sorted(list(bams_dir.glob("*.bam")))
        if not bams:
            bams = sorted(list(bams_dir.glob("*/*.bam")))

        valid_bams = [b for b in bams if "candidate" not in b.name.lower() and not b.name.endswith(".qname.bam")]
        if not valid_bams:
            print(f"[ERROR] No valid BAM files found in {bams_dir}.", file=sys.stderr)
            sys.exit(1)

        print(f"\n[*] Discovered {len(valid_bams)} BAM file(s) in: {bams_dir}")
        summary_records = []
        for bam_file in valid_bams:
            sample_name = get_sample_name_from_bam(bam_file)
            target_bed = find_target_bed_for_sample(
                sample_name, target_dir, fallback_bed=Path(args.target_bed) if args.target_bed else None
            )
            if not target_bed:
                print(f"\n[WARN] No matching candidate BED found for BAM '{bam_file.name}' (Sample: {sample_name}).")
                print(f"       Checked in: {target_dir}. Skipping.")
                continue

            rec = process_single_sample(
                sample_name=sample_name,
                bam_file=bam_file,
                target_bed_file=target_bed,
                reference_file=ref_path,
                output_dir=out_dir,
                args=args,
            )
            if rec:
                summary_records.append(rec)

        if summary_records:
            export_batch_cohort_summary(summary_records, out_dir, args)
        return

    # 模式 C: 自动扫描 FASTQ 并比对
    input_dir = Path(args.input_dir).resolve()
    bam_dir = out_dir / "bwa_bams"
    bam_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Scanning for paired FASTQ files in: {input_dir}")
    pairs = find_fastq_pairs(input_dir)
    if not pairs:
        print(f"[ERROR] No paired FASTQ files found in {input_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Discovered {len(pairs)} sample pair(s) to process:")
    summary_records = []

    for sample_name, r1, r2 in pairs:
        target_bed = find_target_bed_for_sample(
            sample_name, target_dir, fallback_bed=Path(args.target_bed) if args.target_bed else None
        )
        if not target_bed:
            print(f"\n[WARN] No matching candidate BED found for sample '{sample_name}'. Skipping.")
            continue

        sample_bam = bam_dir / f"{sample_name}_bwa.sorted.bam"
        try:
            aligned_bam = align_and_sort_bwa(
                r1=r1,
                r2=r2,
                reference=ref_path,
                output_bam=sample_bam,
                threads=args.threads,
                force_align=args.force_align,
            )
        except Exception as e:
            print(f"[ERROR] BWA alignment failed for {sample_name}: {e}", file=sys.stderr)
            continue

        rec = process_single_sample(
            sample_name=sample_name,
            bam_file=aligned_bam,
            target_bed_file=target_bed,
            reference_file=ref_path,
            output_dir=out_dir,
            args=args,
        )
        if rec:
            summary_records.append(rec)

    if summary_records:
        export_batch_cohort_summary(summary_records, out_dir, args)


if __name__ == "__main__":
    main()
