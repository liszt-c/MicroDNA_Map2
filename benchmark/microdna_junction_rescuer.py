#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MicroDNA Junction Read Rescuer (Automated BWA Pipeline & Anchor-Constrained Rescuer)
===================================================================================
功能:
  1. 自动化模式: 自动扫描 data/raw 下的双端 FASTQ 文件，调用 BWA-MEM 进行流式比对与排序索引，
     并自动匹配 results/predictions 下的候选微环 BED 进行接头读段 (Junction Read) 挽救评估。
  2. 独立模式: 兼容直接传入已比对好的 BAM (-b) 与候选 BED (-t) 进行单样本分析。
  3. 证据模型: 结合断裂点锚定 (Anchor Matching) 的 Split-Reads 与 Everted-Pairs 双模态证据，
     通过协变量匹配对照 (Matched Controls) 与 Fisher/Mann-Whitney 检验评估统计特异性。
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
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
        description="Automated BWA alignment & Circular DNA Junction Read Rescuer Pipeline."
    )
    # 模式选择与数据输入
    parser.add_argument("-b", "--bam", default=None,
                        help="Path to an existing BAM file (disables automated FASTQ alignment if set)")
    parser.add_argument("-t", "--target-bed", default=None,
                        help="Target candidate BED file (or fallback BED in batch mode)")
    parser.add_argument("--auto", action="store_true",
                        help="Force automated discovery and alignment of FASTQs in --input-dir")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RAW_DIR,
                        help=f"Directory containing paired FASTQ files (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_PRED_DIR,
                        help=f"Directory containing predicted candidate BED files (default: {DEFAULT_PRED_DIR})")
    parser.add_argument("-r", "--reference", type=Path, default=DEFAULT_REF,
                        help=f"Reference genome FASTA file (default: {DEFAULT_REF})")
    parser.add_argument("--bam-dir", type=Path, default=None,
                        help="Directory to save/load BWA-aligned BAMs (default: <output-dir>/bwa_bams)")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory for rescue reports (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("-p", "--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="CPU threads for BWA and Samtools (default: max available - 1)")
    parser.add_argument("--force-align", action="store_true",
                        help="Force BWA re-alignment even if sorted BAM already exists")

    # 接头捕获与统计参数
    parser.add_argument("--mode", choices=["all", "split", "everted"], default="all",
                        help="Evidence capture mode: 'split' (SA tag only), 'everted' (RF mate-pairs only), or 'all' (default: all)")
    parser.add_argument("--anchor-tol", type=int, default=50,
                        help="Breakpoint anchor matching tolerance in bp (default: 50 bp)")
    parser.add_argument("--min-mapq", type=int, default=10,
                        help="Minimum MAPQ quality threshold (default: 10)")
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
# 1. FASTQ 配对自动发现与比对准备模块
# =====================================================================

def find_fastq_pairs(directory: Path) -> List[Tuple[str, Path, Path]]:
    """
    智能扫描目录下的配对 FASTQ 文件，支持 .fastq, .fq 及其 .gz 压缩格式
    返回列表: [(sample_name, r1_path, r2_path), ...]
    """
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
    """确保参考基因组的 BWA 索引与 Samtools faidx 索引存在，缺失则自动构建"""
    reference = Path(reference).resolve()
    if not reference.exists():
        raise FileNotFoundError(f"Reference genome FASTA not found: {reference}")

    # 1. 检查 BWA 索引
    bwa_exts = [".amb", ".ann", ".bwt", ".pac", ".sa"]
    has_bwa = all(Path(str(reference) + ext).exists() for ext in bwa_exts)
    if not has_bwa:
        if shutil.which("bwa") is None:
            raise RuntimeError("Tool 'bwa' not found in PATH. Please install bwa.")
        print(f"[*] BWA index missing. Generating BWA index for {reference.name} (this may take several minutes)...")
        subprocess.run(["bwa", "index", str(reference)], check=True)
        print("[+] BWA index built successfully.")
    else:
        print(f"[+] BWA index verified for {reference.name}.")

    # 2. 检查 faidx 索引
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
    """使用 BWA-MEM 将配对 FASTQ 比对并直接管道排序输出 BAM 与 BAI 索引"""
    output_bam = Path(output_bam).resolve()
    output_bam.parent.mkdir(parents=True, exist_ok=True)

    # 复用检测
    if not force_align and output_bam.exists() and output_bam.stat().st_size > 0:
        bai = Path(str(output_bam) + ".bai")
        if not bai.exists():
            subprocess.run(["samtools", "index", "-@", str(threads), str(output_bam)], check=True)
        print(f"[*] Found existing valid BWA BAM: {output_bam.name}, skipping alignment.")
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

    print(f"[*] Indexing BAM: {output_bam.name}...")
    subprocess.run(["samtools", "index", "-@", str(threads), str(output_bam)], check=True)
    print(f"[+] BWA alignment completed -> {output_bam}")
    return output_bam


def find_target_bed_for_sample(
    sample_name: str,
    target_dir: Path,
    fallback_bed: Optional[Path] = None,
) -> Optional[Path]:
    """为特定样本自动匹配其预测的 candidate BED 文件"""
    if target_dir.exists():
        candidates = [
            target_dir / f"{sample_name}_microDNA.bed",
            target_dir / f"{sample_name}.bed",
            target_dir / f"{sample_name}_predictions.bed",
            target_dir / f"{sample_name}_candidates.bed",
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                return c

        fuzzy = list(target_dir.glob(f"{sample_name}*.bed"))
        if fuzzy:
            return fuzzy[0]

    if fallback_bed and Path(fallback_bed).exists():
        return Path(fallback_bed)

    return None


# =====================================================================
# 2. 染色体名称标准化与区域加载
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


def get_region_mean_depth(bam: pysam.AlignmentFile, chrom: str, start: int, end: int) -> float:
    try:
        s = max(0, start)
        e = max(s + 1, end)
        cov = bam.count_coverage(chrom, s, e)
        total_bases = sum(sum(cov[i]) for i in range(4))
        return total_bases / max(1, e - s)
    except Exception:
        return 0.0


def generate_matched_controls(
    targets: List[Dict],
    bam: pysam.AlignmentFile,
    chrom_lens: Dict[str, int],
    seed: int = 42,
    depth_tol: float = 0.35,
) -> List[Dict]:
    random.seed(seed)
    forbidden = defaultdict(list)
    for t in targets:
        forbidden[t["chrom"]].append((max(0, t["start"] - 1500), t["end"] + 1500))

    controls = []
    print(f"[*] Generating matched controls for {len(targets)} candidate regions...")

    for t in targets:
        chrom = t["chrom"]
        if chrom not in chrom_lens:
            continue
        c_len = chrom_lens[chrom]
        r_len = t["len"]
        target_depth = get_region_mean_depth(bam, chrom, t["start"], t["end"])

        best_ctrl = None
        min_depth = target_depth * (1.0 - depth_tol)
        max_depth = target_depth * (1.0 + depth_tol)

        max_start = max(10001, c_len - r_len - 10000)
        if max_start <= 10000:
            continue

        for _ in range(150):
            cand_start = random.randint(10000, max_start)
            cand_end = cand_start + r_len

            # 避让重叠检测
            if any(cs < cand_end and cand_start < ce for cs, ce in forbidden[chrom]):
                continue

            # 测序深度匹配检测
            if target_depth > 1.0:
                c_depth = get_region_mean_depth(bam, chrom, cand_start, cand_end)
                if not (min_depth <= c_depth <= max_depth):
                    continue

            best_ctrl = {
                "chrom": chrom,
                "raw_chrom": t.get("raw_chrom", chrom),
                "start": cand_start,
                "end": cand_end,
                "id": f"CTRL_{t['id']}",
                "len": r_len,
            }
            forbidden[chrom].append((max(0, cand_start - 1500), cand_end + 1500))
            break

        if best_ctrl is None:
            cand_start = random.randint(10000, max_start)
            best_ctrl = {
                "chrom": chrom,
                "raw_chrom": t.get("raw_chrom", chrom),
                "start": cand_start,
                "end": cand_start + r_len,
                "id": f"CTRL_{t['id']}",
                "len": r_len,
            }

        controls.append(best_ctrl)

    return controls


# =====================================================================
# 3. 严谨锚定接头证据判定引擎
# =====================================================================

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
    reg_len: int,
    bam_chroms: Set[str],
    anchor_tol: int = 50,
    min_mapq: int = 10,
    min_len: int = 150,
    max_len: int = 3000,
) -> Optional[Dict]:
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
        if sa["mapq"] < min_mapq:
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

        # 1. 严格断裂点边界锚定校验
        if abs(junc_left - reg_start) > anchor_tol or abs(junc_right - reg_end) > anchor_tol:
            continue
        if abs(circle_len - reg_len) > anchor_tol * 2:
            continue

        # 2. 几何拓扑校验 (正链/负链)
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
    reg_len: int,
    bam_chroms: Set[str],
    anchor_tol: int = 50,
    min_mapq: int = 10,
    min_len: int = 150,
    max_len: int = 3000,
) -> Optional[Dict]:
    if not read.is_paired or read.is_unmapped or read.mate_is_unmapped:
        return None
    # 核心过滤: 排除正常线性配对 (消除全基因组随机背景噪声)
    if read.is_proper_pair:
        return None
    if read.mapping_quality < min_mapq:
        return None
    if read.reference_name != read.next_reference_name:
        return None

    chrom = read.reference_name
    r_start = read.reference_start
    r_end = read.reference_end
    m_start = read.next_reference_start

    # 情况 A: 当前 read 为正链 (+)，mate 为负链 (-)
    if not read.is_reverse and read.mate_is_reverse:
        if r_start > m_start:
            span = (r_end if r_end else r_start + 150) - m_start
            if not (min_len <= span <= max_len):
                return None
            if abs(m_start - reg_start) <= anchor_tol + 50 and abs(r_start - reg_end) <= anchor_tol + 150:
                if abs(span - reg_len) <= anchor_tol * 2 + 150:
                    return {
                        "qname": read.query_name,
                        "evidence_type": "Everted-Pair",
                        "chrom": chrom,
                        "junction_start": m_start,
                        "junction_end": r_end if r_end else r_start + 150,
                        "circle_len": span,
                        "prim_coords": f"{r_start}-{r_end}",
                        "split_coords": f"mate:{m_start}",
                        "mapq": f"{read.mapping_quality}",
                    }

    # 情况 B: 当前 read 为负链 (-)，mate 为正链 (+)
    elif read.is_reverse and not read.mate_is_reverse:
        if m_start > r_start:
            span = (m_start + 150) - r_start
            if not (min_len <= span <= max_len):
                return None
            if abs(r_start - reg_start) <= anchor_tol + 50 and abs(m_start - reg_end) <= anchor_tol + 150:
                if abs(span - reg_len) <= anchor_tol * 2 + 150:
                    return {
                        "qname": read.query_name,
                        "evidence_type": "Everted-Pair",
                        "chrom": chrom,
                        "junction_start": r_start,
                        "junction_end": m_start + 150,
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


def scan_cohort_regions(
    regions: List[Dict],
    bam: pysam.AlignmentFile,
    ref_fasta: pysam.FastaFile,
    group_type: str,
    args,
    bam_chroms: Set[str],
) -> Tuple[List[Dict], List[Dict], int]:
    region_records = []
    all_rescued_reads = []
    sa_tags_seen = 0

    for reg in regions:
        chrom = reg["chrom"]
        start = reg["start"]
        end = reg["end"]
        reg_len = reg["len"]

        search_start = max(0, start - max(150, args.anchor_tol))
        search_end = end + max(150, args.anchor_tol)

        rescued_in_this_region = {}
        mean_depth = get_region_mean_depth(bam, chrom, start, end)

        try:
            for read in bam.fetch(chrom, search_start, search_end):
                if read.is_unmapped or read.is_duplicate:
                    continue

                if read.has_tag("SA"):
                    sa_tags_seen += 1

                junc_info = None

                # 1. 尝试匹配 Split Read
                if args.mode in ["all", "split"]:
                    junc_info = detect_head_to_tail_split_read(
                        read, start, end, reg_len, bam_chroms,
                        anchor_tol=args.anchor_tol, min_mapq=args.min_mapq,
                        min_len=args.min_circle_len, max_len=args.max_circle_len
                    )

                # 2. 尝试匹配 Everted Pair
                if not junc_info and args.mode in ["all", "everted"]:
                    junc_info = detect_everted_mate_pair(
                        read, start, end, reg_len, bam_chroms,
                        anchor_tol=args.anchor_tol, min_mapq=args.min_mapq,
                        min_len=args.min_circle_len, max_len=args.max_circle_len
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
        all_rescued_reads.extend(rescued_in_this_region.values())

    return region_records, all_rescued_reads, sa_tags_seen


# =====================================================================
# 4. 单样本接头挽救评测工作流
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

    print(f"[*] Scanning Target ROIs (n={len(targets)}) [Anchor Tol: ±{args.anchor_tol}bp]...")
    target_summary, target_junctions, target_sa_count = scan_cohort_regions(
        targets, bam, ref_fasta, "Target", args, bam_chroms
    )

    print(f"[*] Scanning Control ROIs (n={len(controls)}) [Anchor Tol: ±{args.anchor_tol}bp]...")
    control_summary, control_junctions, control_sa_count = scan_cohort_regions(
        controls, bam, ref_fasta, "Control", args, bam_chroms
    )

    # 导出明细 TSV
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

    # 统计检验
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
    print(f"[*] Target Breakdown: {split_cnt} Split-Reads, {evert_cnt} Everted-Pairs")
    print(f"[*] Fisher's Exact Test Odds Ratio (OR): {odds_ratio:.3f}")
    print(f"[*] Fisher's Exact Test p-value:         {p_value:.3e}")
    print(f"[*] Mann-Whitney U Test p-value:         {u_pval:.3e}")
    if split_cnt > 0:
        print(f"[*] Average Microhomology (Split-reads): {avg_hom:.2f} bp")
    print("-" * 70 + "\n")

    bam.close()
    ref_fasta.close()

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
# 5. 全流程自动化总控 (Main Entry)
# =====================================================================

def main():
    args = parse_args()
    ref_path = Path(args.reference).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bam_dir = Path(args.bam_dir).resolve() if args.bam_dir else out_dir / "bwa_bams"
    bam_dir.mkdir(parents=True, exist_ok=True)

    # 1. 确保参考基因组及其索引就绪
    ensure_reference_indices(ref_path)

    # 2. 判断运行模式: 单 BAM 模式 vs 自动 FASTQ 发现比对模式
    if args.bam is not None:
        # 单 BAM 模式
        bam_file = Path(args.bam).resolve()
        if not bam_file.exists():
            print(f"[ERROR] Specified BAM file does not exist: {bam_file}", file=sys.stderr)
            sys.exit(1)
        if not args.target_bed:
            print("[ERROR] Single BAM mode requires '--target-bed' argument.", file=sys.stderr)
            sys.exit(1)

        sample_name = bam_file.stem.replace(".sorted", "").replace("_bwa", "")
        target_bed = Path(args.target_bed).resolve()
        process_single_sample(sample_name, bam_file, target_bed, ref_path, out_dir, args)
        return

    # 自动化模式: 扫描 --input-dir 下的双端测序数据
    input_dir = Path(args.input_dir).resolve()
    target_dir = Path(args.target_dir).resolve()

    print(f"\n[*] Scanning for paired FASTQ files in: {input_dir}")
    pairs = find_fastq_pairs(input_dir)

    if not pairs:
        print(f"[ERROR] No paired FASTQ files found in {input_dir}.", file=sys.stderr)
        print("  - Expected file patterns: *_1.fastq / *_2.fastq, *_R1.fq.gz / *_R2.fq.gz, etc.")
        sys.exit(1)

    print(f"[+] Discovered {len(pairs)} sample pair(s) to process:")
    for s_name, r1, r2 in pairs:
        print(f"    - {s_name:<20}: {r1.name} & {r2.name}")

    summary_records = []

    for sample_name, r1, r2 in pairs:
        # 匹配候选 BED
        target_bed = find_target_bed_for_sample(
            sample_name, target_dir, fallback_bed=Path(args.target_bed) if args.target_bed else None
        )
        if not target_bed:
            print(f"\n[WARN] No matching candidate BED found for sample '{sample_name}'.")
            print(f"       Checked under: {target_dir}")
            print(f"       Skipping junction rescue for '{sample_name}'.")
            continue

        # 执行/复用 BWA 比对
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

        # 执行接头挽救分析
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

    # 导出批处理汇总报表
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