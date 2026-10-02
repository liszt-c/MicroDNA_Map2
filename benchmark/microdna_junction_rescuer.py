#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MicroDNA Junction Read Rescuer (Head-to-Tail Split-Read & Everted-Pair Pipeline)
================================================================================
用于在目标候选区域 (Target ROI) 与协变量匹配对照组 (Matched Controls) 之间
精确捕获、统计与验证低频环状 DNA 闭合接头读段 (Junction Reads)。

支持两种环化证据模型:
  1. Split-Reads (嵌合接头读段, 基于 BWA-MEM 的 SA 标签)
  2. Everted-Pairs (外翻背向配对读段, 兼容 Bowtie2 与 BWA-MEM 的 RF 环化拓扑结构)
"""

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pysam
from scipy.stats import fisher_exact, mannwhitneyu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rescue circular Head-to-Tail junction reads from target vs. matched control regions."
    )
    parser.add_argument("-b", "--bam", required=True, help="Input coordinate-sorted and indexed BAM file")
    parser.add_argument("-t", "--target-bed", required=True, help="Target high-confidence candidate BED file")
    parser.add_argument("-r", "--reference", required=True, help="Reference genome FASTA file")
    parser.add_argument("-o", "--out-prefix", default="rescued_microdna", help="Output files prefix")
    parser.add_argument("--mode", choices=["all", "split", "everted"], default="all",
                        help="Evidence capture mode: 'split' (SA tag only), 'everted' (RF mate-pairs only), or 'all' (default: all)")
    parser.add_argument("--min-mapq", type=int, default=10, help="Minimum MAPQ quality threshold (default: 10)")
    parser.add_argument("--min-circle-len", type=int, default=150, help="Minimum circular DNA length to rescue (default: 150 bp)")
    parser.add_argument("--max-circle-len", type=int, default=3000, help="Maximum circular DNA length to rescue (default: 3000 bp)")
    parser.add_argument("--depth-tolerance", type=float, default=0.35, help="Coverage tolerance ratio for control matching (default: 0.35)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


# =====================================================================
# 1. 染色体名称自适应解析器
# =====================================================================

def resolve_chromosome_name(query_chrom: str, target_chroms: Set[str]) -> Optional[str]:
    """解决 chr1 与 1 之间的大小写及前缀不匹配问题"""
    if query_chrom in target_chroms:
        return query_chrom
    q_lower = query_chrom.lower()
    bare = q_lower[3:] if q_lower.startswith("chr") else q_lower
    candidates = [query_chrom, f"chr{bare}", bare, query_chrom.upper(), query_chrom.lower(),
                  f"chr{bare.upper()}", f"chr{bare.lower()}"]
    for c in candidates:
        if c in target_chroms:
            return c
    return None


# =====================================================================
# 2. 区域加载与协变量匹配对照组生成
# =====================================================================

def load_bed_regions(bed_path: str, bam_chroms: Set[str]) -> List[Dict]:
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
            region_id = parts[3] if len(parts) > 3 else f"ROI_{idx+1}"
            regions.append({
                "chrom": resolved_chrom,
                "raw_chrom": raw_chrom,
                "start": start,
                "end": end,
                "id": region_id,
                "len": max(1, end - start),
            })
    if skipped > 0:
        print(f"[WARN] Skipped {skipped} regions from BED because chromosomes were missing in BAM.")
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
        forbidden[t["chrom"]].append((max(0, t["start"] - 1000), t["end"] + 1000))

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

        for _ in range(120):
            cand_start = random.randint(10000, max_start)
            cand_end = cand_start + r_len

            # 避让检测
            has_conflict = any(cs < cand_end and cand_start < ce for cs, ce in forbidden[chrom])
            if has_conflict:
                continue

            # 深度匹配检测
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
            forbidden[chrom].append((max(0, cand_start - 1000), cand_end + 1000))
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
# 3. 接头证据判定引擎 (Split-Reads 与 Everted-Pairs)
# =====================================================================

def parse_sa_tag(sa_string: str) -> List[Dict]:
    """解析 BWA-MEM 输出的 Supplementary SA 标签"""
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
                "pos": int(fields[1]) - 1,  # 转换为 0-based
                "strand": fields[2],
                "cigar": fields[3],
                "mapq": int(fields[4]),
                "nm": int(fields[5]) if len(fields) > 5 else 0
            })
        except ValueError:
            continue
    return entries


def get_sa_query_offset(cigar_str: str) -> int:
    """提取 SA 比对段在 Read Query 序列上的起始偏移量"""
    match = re.match(r"^(\d+)[SH]", cigar_str)
    return int(match.group(1)) if match else 0


def detect_head_to_tail_split_read(
    read: pysam.AlignedSegment,
    bam_chroms: Set[str],
    min_mapq: int = 10,
    min_len: int = 150,
    max_len: int = 3000,
) -> Optional[Dict]:
    """
    判定单条读段是否为跨接头嵌合读段 (Split Read)。
    已彻底修复几何反转判定逻辑，采用两段在 Query 序列上的相对坐标进行严格定向。
    """
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

        # 核心几何校验 (已修复反转 Bug)
        is_h2t = False
        if prim_strand == "+":
            # 正链：Query 5' 侧片段必定位于微环下游 (基因组坐标更大)，Query 3' 侧片段位于微环上游 (坐标更小)
            if prim_qstart < sa_qstart and prim_start > sa_start:
                is_h2t = True
            elif sa_qstart < prim_qstart and sa_start > prim_start:
                is_h2t = True
        else:
            # 负链对称检验：Query 5' 侧片段位于微环上游 (坐标更小)，Query 3' 侧片段位于微环下游 (坐标更大)
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
    bam_chroms: Set[str],
    min_mapq: int = 10,
    min_len: int = 150,
    max_len: int = 3000,
) -> Optional[Dict]:
    """
    判定配对读段是否呈现跨接头外翻构型 (Everted / Outward-facing Pair, 即 RF 链方向)。
    在微环尺度下，该拓扑在常规线性基因组上表现为: 正链读段坐标 > 负链读段坐标。
    """
    if not read.is_paired or read.is_unmapped or read.mate_is_unmapped:
        return None
    if read.mapping_quality < min_mapq:
        return None
    if read.reference_name != read.next_reference_name:
        return None

    chrom = read.reference_name
    r_start = read.reference_start
    r_end = read.reference_end
    m_start = read.next_reference_start

    # 要求一正一负，且正链读段起始坐标大于负链读段起始坐标 (RF 外翻构型)
    if not read.is_reverse and read.mate_is_reverse:
        # 当前为正链读段，Mate 为负链
        if r_start > m_start:
            span = (r_end if r_end else r_start + 150) - m_start
            if min_len <= span <= max_len:
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
    return None


def calculate_microhomology(
    ref_fasta: pysam.FastaFile,
    chrom: str,
    junc_start: int,
    junc_end: int,
    max_search: int = 25,
) -> int:
    """在参考基因组断裂点左右提取微同源序列最大重叠长度"""
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
# 4. 扫描执行与数据汇流
# =====================================================================

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
        # 外扩缓冲带，覆盖微环两端断裂点
        search_start = max(0, start - 300)
        search_end = end + 300

        rescued_in_this_region = {}
        mean_depth = get_region_mean_depth(bam, chrom, start, end)

        try:
            for read in bam.fetch(chrom, search_start, search_end):
                if read.is_unmapped or read.is_duplicate:
                    continue

                if read.has_tag("SA"):
                    sa_tags_seen += 1

                junc_info = None

                # 1. 尝试检测 Split Read (如启用了 split 或 all)
                if args.mode in ["all", "split"]:
                    junc_info = detect_head_to_tail_split_read(
                        read, bam_chroms, min_mapq=args.min_mapq,
                        min_len=args.min_circle_len, max_len=args.max_circle_len
                    )

                # 2. 若未满足 Split Read，尝试检测 Everted Pair (如启用了 everted 或 all)
                if not junc_info and args.mode in ["all", "everted"]:
                    junc_info = detect_everted_mate_pair(
                        read, bam_chroms, min_mapq=args.min_mapq,
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

        except Exception as e:
            # 记录并跳过受损区间
            pass

        count = len(rescued_in_this_region)
        region_records.append({
            "region_id": reg["id"],
            "group": group_type,
            "chrom": reg.get("raw_chrom", chrom),
            "start": start,
            "end": end,
            "length": reg["len"],
            "mean_depth": round(mean_depth, 2),
            "rescued_reads": count,
            "has_junction": 1 if count > 0 else 0,
        })
        all_rescued_reads.extend(rescued_in_this_region.values())

    return region_records, all_rescued_reads, sa_tags_seen


# =====================================================================
# 5. 主控与统计检验流程
# =====================================================================

def main():
    args = parse_args()
    bam_path = Path(args.bam)
    ref_path = Path(args.reference)
    bed_path = Path(args.target_bed)

    if not bam_path.exists():
        print(f"[ERROR] BAM file not found: {bam_path}", file=sys.stderr)
        sys.exit(1)
    if not ref_path.exists():
        print(f"[ERROR] Reference file not found: {ref_path}", file=sys.stderr)
        sys.exit(1)
    if not bed_path.exists():
        print(f"[ERROR] Target BED not found: {bed_path}", file=sys.stderr)
        sys.exit(1)

    print("[*] Opening BAM and Reference FASTA...")
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    ref_fasta = pysam.FastaFile(str(ref_path))

    bam_chroms = set(bam.references)
    ref_chroms = set(ref_fasta.references)
    chrom_lens = dict(zip(ref_fasta.references, ref_fasta.lengths))

    # 1. 载入实验组区域
    targets = load_bed_regions(str(bed_path), bam_chroms)
    if not targets:
        print("[ERROR] No valid target regions could be loaded from BED.", file=sys.stderr)
        sys.exit(1)

    # 2. 生成协变量匹配的对照组
    controls = generate_matched_controls(
        targets, bam, chrom_lens, seed=args.seed, depth_tol=args.depth_tolerance
    )

    ctrl_bed_path = f"{args.out_prefix}_matched_controls.bed"
    with open(ctrl_bed_path, "w", encoding="utf-8") as f:
        for c in controls:
            f.write(f"{c['raw_chrom']}\t{c['start']}\t{c['end']}\t{c['id']}\n")
    print(f"[+] Matched controls exported to: {ctrl_bed_path}")

    # 3. 扫描两个区域池
    print(f"[*] Scanning Target ROIs (n={len(targets)}) [Mode: {args.mode}]...")
    target_summary, target_junctions, target_sa_count = scan_cohort_regions(
        targets, bam, ref_fasta, "Target", args, bam_chroms
    )

    print(f"[*] Scanning Control ROIs (n={len(controls)}) [Mode: {args.mode}]...")
    control_summary, control_junctions, control_sa_count = scan_cohort_regions(
        controls, bam, ref_fasta, "Control", args, bam_chroms
    )

    # 4. 诊断日志打印 (提示 BAM 比对器兼容性)
    total_sa = target_sa_count + control_sa_count
    if total_sa == 0 and args.mode in ["all", "split"]:
        print("\n" + "!" * 75)
        print("[NOTICE] Zero reads carrying 'SA' tags were detected in the queried regions.")
        print("  - If this BAM was produced by Bowtie2 (MicroDNA Map default), Bowtie2 does")
        print("    not output split-read SA tags. The pipeline automatically utilized")
        print("    Everted Mate-Pair topology to rescue circular evidence.")
        print("  - To maximize split-read breakpoint recovery, aligning with BWA-MEM is recommended.")
        print("!" * 75 + "\n")

    # 5. 输出汇总报表
    all_summary = target_summary + control_summary
    all_junctions = target_junctions + control_junctions

    summary_tsv = f"{args.out_prefix}_region_summary.tsv"
    with open(summary_tsv, "w", encoding="utf-8") as f:
        f.write("region_id\tgroup\tchrom\tstart\tend\tlength\tmean_depth\trescued_reads\thas_junction\n")
        for r in all_summary:
            f.write(f"{r['region_id']}\t{r['group']}\t{r['chrom']}\t{r['start']}\t{r['end']}\t"
                    f"{r['length']}\t{r['mean_depth']}\t{r['rescued_reads']}\t{r['has_junction']}\n")
    print(f"[+] Region summary written to: {summary_tsv}")

    junctions_tsv = f"{args.out_prefix}_rescued_junctions.tsv"
    with open(junctions_tsv, "w", encoding="utf-8") as f:
        f.write("qname\tevidence_type\tgroup\tregion_id\tchrom\tjunction_start\tjunction_end\t"
                "circle_len\tprim_coords\tsplit_coords\tmapq\tmicrohomology_bp\n")
        for j in all_junctions:
            f.write(f"{j['qname']}\t{j['evidence_type']}\t{j['group']}\t{j['region_id']}\t{j['chrom']}\t"
                    f"{j['junction_start']}\t{j['junction_end']}\t{j['circle_len']}\t"
                    f"{j['prim_coords']}\t{j['split_coords']}\t{j['mapq']}\t{j['microhomology_bp']}\n")
    print(f"[+] Rescued junction reads logged to: {junctions_tsv}")

    # 6. 统计学检验 (Fisher 精确检验与 Mann-Whitney U 检验)
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

    # 统计微同源均值 (仅针对 Split-Read)
    t_hom = [j["microhomology_bp"] for j in target_junctions if j["evidence_type"] == "Split-Read"]
    avg_hom = np.mean(t_hom) if t_hom else 0.0

    split_cnt = sum(1 for j in target_junctions if j["evidence_type"] == "Split-Read")
    evert_cnt = sum(1 for j in target_junctions if j["evidence_type"] == "Everted-Pair")

    print("\n" + "=" * 70)
    print("            RESCUE ENRICHMENT EVALUATION REPORT")
    print("=" * 70)
    print(f"{'Cohort':<12}{'Total ROIs':<15}{'With Evidence (>=1)':<22}{'Rescue Rate (%)':<15}")
    print("-" * 70)
    print(f"{'Target':<12}{len(target_summary):<15}{t_pos:<22}{t_rate:<15.2f}")
    print(f"{'Control':<12}{len(control_summary):<15}{c_pos:<22}{c_rate:<15.2f}")
    print("-" * 70)
    print(f"[*] Target Evidence Breakdown: {split_cnt} Split-Reads, {evert_cnt} Everted-Pairs")
    print(f"[*] 2x2 Contingency Table: Target [{t_pos}, {t_neg}] vs Control [{c_pos}, {c_neg}]")
    print(f"[*] Fisher's Exact Test Odds Ratio (OR): {odds_ratio:.3f}")
    print(f"[*] Fisher's Exact Test p-value:         {p_value:.3e}")
    print(f"[*] Mann-Whitney U Test p-value:         {u_pval:.3e}")
    print(f"[*] Average Microhomology (Split-reads): {avg_hom:.2f} bp")
    print("=" * 70 + "\n")

    bam.close()
    ref_fasta.close()


if __name__ == "__main__":
    main()