#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MicroDNA Junction Read Rescuer
===================================================================
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
import pysam
from scipy.stats import fisher_exact, mannwhitneyu

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
        description="MicroDNA Junction Read Rescuer (Multi-Core High-Throughput Pipeline)"
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
                        help="CPU threads for BWA, Samtools, and parallel region scanning (default: max - 1)")
    parser.add_argument("--force-align", action="store_true",
                        help="Force BWA re-alignment even if output BAM already exists")

    # 算法与证据判定参数
    parser.add_argument("--mode", choices=["all", "split", "everted"], default="all",
                        help="Evidence capture mode: 'split', 'everted', or 'all' (default: all)")
    parser.add_argument("--min-overlap", type=float, default=0.40,
                        help="Minimum reciprocal overlap ratio between candidate region and rescued circle (default: 0.40)")
    parser.add_argument("--flank-tol", type=int, default=150,
                        help="Maximum flank extension tolerance in bp for boundary containment (default: 150 bp)")
    parser.add_argument("--min-mapq", type=int, default=10,
                        help="Minimum MAPQ quality threshold for primary alignment (default: 10)")
    parser.add_argument("--min-circle-len", type=int, default=150,
                        help="Minimum circular DNA length to rescue (default: 150 bp)")
    parser.add_argument("--max-circle-len", type=int, default=3000,
                        help="Maximum circular DNA length to rescue (default: 3000 bp)")
    parser.add_argument("--depth-tolerance", type=float, default=0.35,
                        help="Coverage tolerance ratio for control matching (default: 0.35)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


# =====================================================================
# 1. 样本名称解析与文件关联检索模块
# =====================================================================

def ensure_bam_indexed(bam_path: Path, threads: int = 4) -> None:
    """确保 BAM 文件具有有效的 .bai 索引，缺失则自动建立"""
    bai1 = Path(str(bam_path) + ".bai")
    bai2 = bam_path.with_suffix(".bai")
    if not (bai1.exists() or bai2.exists()):
        print(f"[*] BAM index missing. Indexing {bam_path.name}...")
        subprocess.run(["samtools", "index", "-@", str(threads), str(bam_path)], check=True)


def get_sample_name_from_bam(bam_path: Path) -> str:
    """
    智能解析 BAM 文件所属样本真实名称。
    按后缀长度从长到短匹配，彻底避免 .sorted.bam 先截断导致残留 _bwa 的 Bug。
    """
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
    """扫描配对 FASTQ 文件，支持 .fastq, .fq 及其 .gz 压缩文件"""
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
    """确保参考基因组的 BWA 索引与 Samtools faidx 存在"""
    reference = Path(reference).resolve()
    if not reference.exists():
        raise FileNotFoundError(f"Reference genome FASTA not found: {reference}")

    bwa_exts = [".amb", ".ann", ".bwt", ".pac", ".sa"]
    has_bwa = all(Path(str(reference) + ext).exists() for ext in bwa_exts)
    if not has_bwa:
        if shutil.which("bwa") is None:
            raise RuntimeError("Tool 'bwa' not found in PATH. Please install bwa.")
        print(f"[*] Generating BWA index for {reference.name} (this may take a few minutes)...")
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
    """使用 BWA-MEM 流式比对并管道排序输出 BAM"""
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
    """为特定样本自动匹配候选预测 BED 文件"""
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
# 2. 染色体名称标准化与极速区间重叠算法
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
    """将重叠/相邻的区间合并为完全不相交的有序区间数组"""
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
    """使用二分检索以 O(log N) 时间复杂度检查区间是否与禁采区冲突"""
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
            regions.append({
                "chrom": resolved_chrom,
                "raw_chrom": raw_chrom,
                "start": start,
                "end": end,
                "id": region_id,
                "len": end - start,
            })
    if skipped > 0:
        print(f"[WARN] Skipped {skipped} regions from BED because chromosomes were absent in BAM.")
    return regions


# =====================================================================
# 3. 极速对照组采样算法 (消除 30 小时 I/O 阻塞)
# =====================================================================

def generate_matched_controls(
    targets: List[Dict],
    bam: pysam.AlignmentFile,
    chrom_lens: Dict[str, int],
    seed: int = 42,
    depth_tol: float = 0.35,
) -> List[Dict]:
    """
    高通量对照组生成器:
    - 利用二分法进行微秒级空间冲突排除。
    - 使用轻量级 bam.count 代替耗时的 count_coverage 进行快速深度校验。
    """
    random.seed(seed)
    raw_forbidden = defaultdict(list)
    for t in targets:
        raw_forbidden[t["chrom"]].append((max(0, t["start"] - 1500), t["end"] + 1500))

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

        # 使用轻量 index count 估算目标区深度
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

        # 尝试 15 次快速抽样
        for _ in range(15):
            cand_start = random.randint(10000, max_start)
            cand_end = cand_start + r_len

            if interval_overlaps_disjoint(merged_ivs, cand_start, cand_end, starts_ivs):
                continue

            if target_depth > 1.0:
                try:
                    c_counts = bam.count(chrom, cand_start, cand_end)
                    c_depth = (c_counts * read_len) / max(1, r_len)
                    # 控制组只要不是深度为 0 的空洞区即可接受匹配
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
            }
            break

        # 回退至几何无冲突随机采样
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
            }

        controls.append(best_ctrl)

    return controls


# =====================================================================
# 4. 稳健拓扑重叠判定引擎 (Overlap-Based Evaluation)
# =====================================================================

def is_concordant_with_region(
    circle_start: int,
    circle_end: int,
    reg_start: int,
    reg_end: int,
    min_overlap: float = 0.40,
    flank_tol: int = 150,
) -> bool:
    """判断检测重建出的微环物理区间是否与候选区域具有拓扑一致性"""
    ov_start = max(circle_start, reg_start)
    ov_end = min(circle_end, reg_end)
    ov_len = max(0, ov_end - ov_start)

    circle_len = max(1, circle_end - circle_start)
    reg_len = max(1, reg_end - reg_start)

    # 1. 严格双向互交重叠率 (防止过大或过小读段错误触发)
    if (ov_len / circle_len >= min_overlap) and (ov_len / reg_len >= min_overlap):
        return True

    # 2. 端点锚定容差 (包容深度学习滑窗激活区的外延漂移)
    if abs(circle_start - reg_start) <= flank_tol and abs(circle_end - reg_end) <= flank_tol:
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
    reg_start: int,
    reg_end: int,
    bam_chroms: Set[str],
    min_overlap: float = 0.40,
    flank_tol: int = 150,
    min_mapq: int = 10,
    min_len: int = 150,
    max_len: int = 3000,
) -> Optional[Dict]:
    """捕获单碱基 Split-Read 并实施拓扑一致性过滤"""
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

        if not is_concordant_with_region(junc_left, junc_right, reg_start, reg_end, min_overlap, flank_tol):
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
    reg_start: int,
    reg_end: int,
    min_overlap: float = 0.40,
    flank_tol: int = 150,
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
    # 使用极速 C 整数对比代替字符串对比
    if read.reference_id != read.next_reference_id:
        return None

    chrom = read.reference_name
    r_start = read.reference_start
    r_end = read.reference_end
    m_start = read.next_reference_start

    # 情况 A: 正链 (+) 在下游，负链 mate (-) 在上游
    if not read.is_reverse and read.mate_is_reverse:
        if r_start > m_start:
            span = (r_end if r_end else r_start + 150) - m_start
            if not (min_len <= span <= max_len):
                return None
            circle_s = m_start
            circle_e = r_end if r_end else r_start + 150
            if is_concordant_with_region(circle_s, circle_e, reg_start, reg_end, min_overlap, flank_tol):
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

    # 情况 B: 负链 (-) 在上游，正链 mate (+) 在下游
    elif read.is_reverse and not read.mate_is_reverse:
        if m_start > r_start:
            span = (m_start + 150) - r_start
            if not (min_len <= span <= max_len):
                return None
            circle_s = r_start
            circle_e = m_start + 150
            if is_concordant_with_region(circle_s, circle_e, reg_start, reg_end, min_overlap, flank_tol):
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
# 5. 高性能多进程区域流式扫描与伴随深度计算
# =====================================================================

def _scan_chunk_worker(
    chunk_regions: List[Dict],
    bam_path: str,
    ref_path: str,
    group_type: str,
    mode: str,
    min_overlap: float,
    flank_tol: int,
    min_mapq: int,
    min_circle_len: int,
    max_circle_len: int,
    bam_chroms: Set[str],
) -> Tuple[List[Dict], List[Dict], int]:
    """单子进程内独立执行的一组区间流式扫描"""
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

        search_start = max(0, start - max(200, flank_tol))
        search_end = end + max(200, flank_tol)

        rescued_in_this_region = {}
        total_overlap_bases = 0

        try:
            for read in bam.fetch(chrom, search_start, search_end):
                if read.is_unmapped or read.is_duplicate:
                    continue

                # 伴随式计算重叠碱基，无需二次调用 count_coverage
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
                        read, start, end, bam_chroms,
                        min_overlap=min_overlap, flank_tol=flank_tol,
                        min_mapq=min_mapq, min_len=min_circle_len,
                        max_len=max_circle_len
                    )

                # 2. 备选捕获 Everted Pair
                if not junc_info and mode in ["all", "everted"]:
                    junc_info = detect_everted_mate_pair(
                        read, start, end,
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
    """基于进程池的多核并行扫描协调器"""
    if not regions:
        return [], [], 0

    workers = min(args.threads, len(regions))
    if workers <= 1:
        return _scan_chunk_worker(
            regions, str(bam_path), str(ref_path), group_type, args.mode,
            args.min_overlap, args.flank_tol, args.min_mapq,
            args.min_circle_len, args.max_circle_len, bam_chroms
        )

    # 切分任务块
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
                args.min_overlap, args.flank_tol, args.min_mapq,
                args.min_circle_len, args.max_circle_len, bam_chroms
            )
            for c in chunks
        ]
        for f in as_completed(futures):
            recs, reads, sa = f.result()
            all_region_records.extend(recs)
            all_rescued_reads.extend(reads)
            total_sa += sa

    # 按原始 ID 恢复顺序
    id_order = {r["id"]: i for i, r in enumerate(regions)}
    all_region_records.sort(key=lambda x: id_order.get(x["region_id"], 0))

    return all_region_records, all_rescued_reads, total_sa


# =====================================================================
# 6. 单样本接头挽救评测工作流
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
    print(f"  - BAM File   : {bam_file}")
    print(f"  - Target BED : {target_bed_file}")

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

    print(f"[*] Scanning Target ROIs (n={len(targets)}) [Threads: {args.threads}, Overlap >= {args.min_overlap:.0%}]...")
    target_summary, target_junctions, target_sa_count = scan_cohort_regions_parallel(
        targets, bam_file, reference_file, "Target", args, bam_chroms
    )

    print(f"[*] Scanning Control ROIs (n={len(controls)}) [Threads: {args.threads}, Overlap >= {args.min_overlap:.0%}]...")
    control_summary, control_junctions, control_sa_count = scan_cohort_regions_parallel(
        controls, bam_file, reference_file, "Control", args, bam_chroms
    )

    all_summary = target_summary + control_summary
    all_junctions = target_junctions + control_junctions

    summary_tsv = f"{sample_out_prefix}_region_summary.tsv"
    with open(summary_tsv, "w", encoding="utf-8") as f:
        f.write("region_id\tgroup\tchrom\tstart\tend\tlength\tmean_depth\trescued_reads\thas_junction\n")
        for r in all_summary:
            f.write(f"{r['region_id']}\t{r['group']}\t{r['chrom']}\t{r['start']}\t{r['end']}\t"
                    f"{r['length']}\t{r['mean_depth']}\t{r['rescued_reads']}\t{r['has_junction']}\n")

    junctions_tsv = f"{sample_out_prefix}_rescued_junctions.tsv"
    with open(junctions_tsv, "w", encoding="utf-8") as f:
        f.write("qname\tevidence_type\tgroup\tregion_id\tchrom\tjunction_start\tjunction_end\t"
                "circle_len\tprim_coords\tsplit_coords\tmapq\tmicrohomology_bp\n")
        for j in all_junctions:
            f.write(f"{j['qname']}\t{j['evidence_type']}\t{j['group']}\t{j['region_id']}\t{j['chrom']}\t"
                    f"{j['junction_start']}\t{j['junction_end']}\t{j['circle_len']}\t"
                    f"{j['prim_coords']}\t{j['split_coords']}\t{j['mapq']}\t{j['microhomology_bp']}\n")

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
    evert_cnt = sum(1 for j in target_junctions if j["evidence_type"] == "Everted-Pair")

    print("\n" + "-" * 70)
    print(f"[{sample_name}] ENRICHMENT EVALUATION REPORT")
    print("-" * 70)
    print(f"{'Cohort':<12}{'Total ROIs':<15}{'With Evidence (>=1)':<22}{'Rescue Rate (%)':<15}")
    print(f"{'Target':<12}{len(target_summary):<15}{t_pos:<22}{t_rate:<15.2f}")
    print(f"{'Control':<12}{len(control_summary):<15}{c_pos:<22}{c_rate:<15.2f}")
    print(f"[*] Target Evidence: {split_cnt} Split-Reads, {evert_cnt} Everted-Pairs")
    print(f"[*] Fisher's Exact Test Odds Ratio (OR): {odds_ratio:.3f}")
    print(f"[*] Fisher's Exact Test p-value:         {p_value:.3e}")
    print(f"[*] Mann-Whitney U Test p-value:         {u_pval:.3e}")
    if split_cnt > 0:
        print(f"[*] Average Microhomology (Split-reads): {avg_hom:.2f} bp")
    print("-" * 70 + "\n")

    return {
        "sample": sample_name,
        "total_targets": len(target_summary),
        "target_rescued": t_pos,
        "target_rate_pct": round(t_rate, 2),
        "control_rescued": c_pos,
        "control_rate_pct": round(c_rate, 2),
        "split_reads": split_cnt,
        "everted_pairs": evert_cnt,
        "odds_ratio": round(odds_ratio, 3),
        "fisher_pval": p_value,
        "mwu_pval": u_pval,
        "avg_microhomology": round(avg_hom, 2),
    }


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

    # -------------------------------------------------------------
    # 模式 A: 单 BAM 显式指定模式 (-b sample.bam)
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 模式 B: BAM 文件夹批量处理模式 (--bam-dir <dir>)
    # -------------------------------------------------------------
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
        for b in valid_bams:
            print(f"    - {b.name}")

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
            batch_summary_tsv = out_dir / "batch_rescue_summary.tsv"
            with open(batch_summary_tsv, "w", encoding="utf-8") as f:
                headers = list(summary_records[0].keys())
                f.write("\t".join(headers) + "\n")
                for r in summary_records:
                    f.write("\t".join(str(r[h]) for h in headers) + "\n")

            print("\n" + "=" * 90)
            print("                 BATCH RESCUE EVALUATION MASTER SUMMARY")
            print("=" * 90)
            print(f"{'Sample':<18}{'Targets':<10}{'Target%':<10}{'Ctrl%':<10}{'Split':<8}{'Everted':<9}{'OR':<8}{'Fisher p':<12}")
            print("-" * 90)
            for r in summary_records:
                print(f"{r['sample']:<18}{r['total_targets']:<10}{r['target_rate_pct']:<10.2f}{r['control_rate_pct']:<10.2f}"
                      f"{r['split_reads']:<8}{r['everted_pairs']:<9}{r['odds_ratio']:<8.2f}{r['fisher_pval']:<12.2e}")
            print("=" * 90)
            print(f"[+] Master summary report saved to: {batch_summary_tsv}\n")
        return

    # -------------------------------------------------------------
    # 模式 C: 自动从 raw FASTQ 目录发现并调用 BWA 进行全自动比对与评估
    # -------------------------------------------------------------
    input_dir = Path(args.input_dir).resolve()
    bam_dir = out_dir / "bwa_bams"
    bam_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Scanning for paired FASTQ files in: {input_dir}")
    pairs = find_fastq_pairs(input_dir)

    if not pairs:
        print(f"[ERROR] No paired FASTQ files found in {input_dir}.", file=sys.stderr)
        print("  - If you already have BAM files, run with '--bam-dir <path>' or '-b <path>'.")
        sys.exit(1)

    print(f"[+] Discovered {len(pairs)} sample pair(s) to process:")
    for s_name, r1, r2 in pairs:
        print(f"    - {s_name:<20}: {r1.name} & {r2.name}")

    summary_records = []

    for sample_name, r1, r2 in pairs:
        target_bed = find_target_bed_for_sample(
            sample_name, target_dir, fallback_bed=Path(args.target_bed) if args.target_bed else None
        )
        if not target_bed:
            print(f"\n[WARN] No matching candidate BED found for sample '{sample_name}'.")
            print(f"       Checked in: {target_dir}. Skipping.")
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
        batch_summary_tsv = out_dir / "batch_rescue_summary.tsv"
        with open(batch_summary_tsv, "w", encoding="utf-8") as f:
            headers = list(summary_records[0].keys())
            f.write("\t".join(headers) + "\n")
            for r in summary_records:
                f.write("\t".join(str(r[h]) for h in headers) + "\n")

        print("\n" + "=" * 90)
        print("                 BATCH RESCUE EVALUATION MASTER SUMMARY")
        print("=" * 90)
        print(f"{'Sample':<18}{'Targets':<10}{'Target%':<10}{'Ctrl%':<10}{'Split':<8}{'Everted':<9}{'OR':<8}{'Fisher p':<12}")
        print("-" * 90)
        for r in summary_records:
            print(f"{r['sample']:<18}{r['total_targets']:<10}{r['target_rate_pct']:<10.2f}{r['control_rate_pct']:<10.2f}"
                  f"{r['split_reads']:<8}{r['everted_pairs']:<9}{r['odds_ratio']:<8.2f}{r['fisher_pval']:<12.2e}")
            print("=" * 90)
            print(f"[+] Master summary report saved to: {batch_summary_tsv}\n")


if __name__ == "__main__":
    main()