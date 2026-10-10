#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_microdna_map.py
===================

封装 MicroDNA Map v2.0 对 WGS / Circle-seq spike-in 数据的检测流程。

支持双后端模式:
  - --pipeline micro_coverage (默认推荐: 自研微尺度局部覆盖度分析)
  - --pipeline cnvkit (基线对照: 原生 CNVkit 流程)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import get_pipeline

DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "detect" / "microdna_map"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEFAULT_LIMIT = "0.75"
DEFAULT_THREADS = 8


def check_tool(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_cmd(cmd: list, cwd: Optional[Path] = None, check: bool = True) -> int:
    print(f"[CMD] {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}")
    return result.returncode


def check_bwa_index(reference: Path) -> bool:
    """检查 BWA 索引文件是否齐全"""
    bwa_exts = [".amb", ".ann", ".bwt", ".pac", ".sa"]
    return all(Path(str(reference) + ext).exists() and Path(str(reference) + ext).stat().st_size > 0 for ext in bwa_exts)


def build_bwa_index(reference: Path) -> None:
    """若 BWA 索引缺失则构建"""
    if check_bwa_index(reference):
        print(f"[INFO] BWA 索引已存在: {reference}")
        return
    print(f"[INFO] 构建 BWA 索引: {reference}")
    run_cmd(["bwa", "index", str(reference)], check=True)


def align_and_sort_reads(r1: Path, r2: Path, reference: Path, output_bam: Path, threads: int) -> None:
    print(f"[INFO] BWA-MEM 比对 reads: {r1.name} & {r2.name}")
    cmd = (
        f"bwa mem -t {threads} {reference} {r1} {r2} | "
        f"samtools sort -@{threads} -o {output_bam}"
    )
    ret = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"BWA 比对失败:\n{ret.stderr}")
    print(f"[INFO] 比对并排序完成: {output_bam}")


def index_bam(bam_path: Path, threads: int) -> None:
    run_cmd(["samtools", "index", "-@", str(threads), str(bam_path)], check=True)


def create_sub_reference(orig_cnn: Path, allowed_chroms: set, output_dir: Path) -> Path:
    """CNVkit 模式专用: 裁剪 .cnn 以匹配模拟染色体子集"""
    sub_cnn_path = output_dir / "sub_reference.cnn"
    if not orig_cnn.exists():
        raise FileNotFoundError(f"Original .cnn not found: {orig_cnn}")

    print(f"[INFO] 正在裁剪 CNVkit 参考基线以匹配局部 WGS 模拟 (保留染色体: {allowed_chroms})...")
    with open(orig_cnn, "r", encoding="utf-8") as fin, open(sub_cnn_path, "w", encoding="utf-8") as fout:
        header = fin.readline()
        fout.write(header)
        h_parts = header.strip().split("\t")
        chrom_idx = h_parts.index("chromosome") if "chromosome" in h_parts else 0
        kept = 0
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) > chrom_idx and parts[chrom_idx] in allowed_chroms:
                fout.write(line)
                kept += 1

    print(f"[INFO] 基线裁剪完成，保留了 {kept} 个 Reference Bins -> {sub_cnn_path.name}")
    return sub_cnn_path


def run_cnvkit(bam_path: Path, cnvkit_reference: Path, output_dir: Path, threads: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "cnvkit.py", "batch", "-m", "wgs", "-r", str(cnvkit_reference),
        "-p", str(threads), "-d", str(output_dir), str(bam_path)
    ], check=True)

    cnr_files = list(output_dir.glob("*.cnr"))
    if not cnr_files:
        raise FileNotFoundError("CNVkit batch 未生成 .cnr 文件")
    cnr_file = cnr_files[0]

    result_cns = output_dir / "result.cns"
    run_cmd(["cnvkit.py", "segment", str(cnr_file), "-p", str(threads), "-m", "cbs", "-o", str(result_cns)], check=True)

    result_call = output_dir / "result.call.cns"
    run_cmd(["cnvkit.py", "call", str(result_cns), "-o", str(result_call)], check=True)
    return result_call


def extract_cnv_regions_fasta(
    call_file: Path,
    reference_fasta: Path,
    output_fa: Path,
    min_length: int = 150,
    allowed_chroms: Optional[set] = None,
) -> int:
    """CNVkit 模式专用序列提取"""
    output_fa.parent.mkdir(parents=True, exist_ok=True)
    if not (reference_fasta.parent / (reference_fasta.name + ".fai")).exists():
        run_cmd(["samtools", "faidx", str(reference_fasta)], check=True)

    lines = []
    with open(call_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if len(lines) < 2:
        output_fa.write_text("")
        return 0

    header = lines[0].split("\t")
    try:
        chrom_idx = header.index("chromosome")
        start_idx = header.index("start")
        end_idx = header.index("end")
    except ValueError:
        chrom_idx, start_idx, end_idx = 0, 1, 2

    regions_to_fetch = []
    for i, line in enumerate(lines[1:]):
        parts = line.split("\t")
        if len(parts) <= max(chrom_idx, start_idx, end_idx):
            continue

        chrom = parts[chrom_idx]
        if allowed_chroms is not None and chrom not in allowed_chroms:
            continue

        try:
            start, end = int(parts[start_idx]), int(parts[end_idx])
        except ValueError:
            continue
        if end - start < min_length:
            continue
        regions_to_fetch.append((chrom, start, end, f"cnv_{i}"))

    from src.utils import extract_regions
    sub_regions = [(c, s, e) for c, s, e, _ in regions_to_fetch]
    seq_map, _ = extract_regions(reference_fasta, sub_regions, "samtools")

    written = 0
    with open(output_fa, "w", encoding="utf-8") as fout:
        for c, s, e, name in regions_to_fetch:
            seq = seq_map.get((c, s, e))
            if not seq:
                continue
            fout.write(f">{name}|{c}:{s}-{e}\n")
            for i in range(0, len(seq), 70):
                fout.write(seq[i : i + 70] + "\n")
            written += 1

    print(f"[INFO] 提取了 {written} 个候选区域序列 -> {output_fa.name}")
    return written


def run_microdna_predict(input_fa: Path, model_path: Path, limit: str, output_dir: Path) -> Optional[Path]:
    predict_script = PROJECT_ROOT / "scripts" / "predict.py"
    if not predict_script.exists():
        raise FileNotFoundError(f"找不到 predict.py: {predict_script}")

    predict_output = output_dir / "predict_results"
    predict_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(predict_script),
        "--input", str(input_fa),
        "--mode", "long",
        "--model", str(model_path),
        "--limit", str(limit),
        "--output-dir", str(predict_output),
    ]

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print("[ERROR] MicroDNA Map predict.py 执行失败")
        return None

    print(f"[INFO] MicroDNA Map 预测完成, 耗时 {elapsed:.2f}s")

    bed_files = list(predict_output.glob("*.bed"))
    if not bed_files:
        print("[WARN] predict.py 未生成任何 BED 文件")
        return None

    if len(bed_files) == 1:
        return bed_files[0]

    merged_bed = predict_output / "merged_predictions.bed"
    with open(merged_bed, "w", encoding="utf-8") as out:
        for bf in bed_files:
            with open(bf, "r", encoding="utf-8") as inp:
                out.write(inp.read())
    return merged_bed


def merge_bed_to_standard(input_bed: Path, output_bed: Path) -> int:
    output_bed.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_bed, "w", encoding="utf-8") as fout:
        if not input_bed.exists():
            return 0
        with open(input_bed, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                chrom, start, end = parts[0], parts[1], parts[2]
                fout.write(f"{chrom}\t{start}\t{end}\tmicrodna_{count}\t1.0\n")
                count += 1
    print(f"[INFO] 标准化 BED 完成, 共 {count} 条记录 -> {output_bed.name}")
    return count


def run_microdna_map(
    r1: Path,
    r2: Path,
    reference: Path,
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    limit: str = DEFAULT_LIMIT,
    threads: int = DEFAULT_THREADS,
    cnvkit_reference: Optional[Path] = None,
    allowed_chroms: Optional[set] = None,
    pipeline_type: str = "micro_coverage",
    fold_change: float = 1.3,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "microdna_map.log"
    total_start = time.time()

    # 1. 统一构建并检查 BWA 索引
    build_bwa_index(reference)

    bam_path = output_dir / "aligned.sorted.bam"
    align_and_sort_reads(r1, r2, reference, bam_path, threads)
    index_bam(bam_path, threads)

    # 2. 候选区域生成
    candidates_fa = output_dir / "candidates.fa"

    if pipeline_type == "micro_coverage":
        logger_pipe = get_pipeline(
            "micro_coverage",
            output_dir=output_dir / "micro_coverage",
            ref_genome=reference,
            fold_change=fold_change,
        )
        result_call = logger_pipe.call_cnv(
            bam_path,
            sample_name="spikein_sample",
            threads=threads,
            allowed_chroms=allowed_chroms,
        )
        candidates = logger_pipe.extract_candidates(result_call, min_size=150, sample_name="spikein")
        n_extracted = logger_pipe.extract_sequences(candidates, candidates_fa)
    else:
        # CNVkit 模式
        cnvkit_dir = output_dir / "cnvkit"
        cnvkit_dir.mkdir(parents=True, exist_ok=True)
        if cnvkit_reference is None:
            cnvkit_reference = PROJECT_ROOT / "refs" / "cnvkit_ref.cnn"
            if not cnvkit_reference.exists():
                raise FileNotFoundError(f"未提供 CNVkit 参考文件: {cnvkit_reference}")

        active_reference = cnvkit_reference
        if allowed_chroms is not None and len(allowed_chroms) > 0:
            active_reference = create_sub_reference(cnvkit_reference, allowed_chroms, cnvkit_dir)

        result_call = run_cnvkit(bam_path, active_reference, cnvkit_dir, threads)
        n_extracted = extract_cnv_regions_fasta(
            result_call, reference, candidates_fa, min_length=150, allowed_chroms=allowed_chroms
        )

    final_bed = output_dir / "detected_microdna.bed"
    if n_extracted == 0:
        final_bed.write_text("")
        return final_bed

    # 3. 滑窗识别
    pred_bed = run_microdna_predict(candidates_fa, model_path, limit, output_dir)
    if pred_bed is None:
        final_bed.write_text("")
        return final_bed

    # 4. 转为标准 BED
    merge_bed_to_standard(pred_bed, final_bed)

    elapsed = time.time() - total_start
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"Pipeline: {pipeline_type}, Total time: {elapsed:.2f}s\n")
    print(f"[INFO] 全部流程完成, 总耗时 {elapsed:.2f} 秒")
    return final_bed


def main() -> None:
    parser = argparse.ArgumentParser(description="MicroDNA Map v2.0 检测封装")
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--r2", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=str, default=DEFAULT_LIMIT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cnvkit_reference", type=Path, default=None)
    parser.add_argument("--allowed_chroms", type=str, default=None, help="逗号分隔的合法染色体列表")
    parser.add_argument(
        "--pipeline",
        choices=["micro_coverage", "cnvkit"],
        default="micro_coverage",
        help="初筛所用的管线后端: micro_coverage (默认) 或 cnvkit",
    )
    parser.add_argument("--fold_change", type=float, default=1.3, help="微尺度分析富集倍数阈值")
    args = parser.parse_args()

    required_tools = ["bwa", "samtools"]
    if args.pipeline == "cnvkit":
        required_tools.append("cnvkit.py")

    for tool in required_tools:
        if not check_tool(tool):
            print(f"[ERROR] 缺少必要工具: {tool}", file=sys.stderr)
            sys.exit(1)

    if not args.r1.exists() or not args.r2.exists() or not args.reference.exists():
        print("[ERROR] 输入文件不存在", file=sys.stderr)
        sys.exit(1)

    if not args.model_path.exists():
        print(f"[ERROR] 模型文件不存在: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    allowed_chroms = set(c.strip() for c in args.allowed_chroms.split(",")) if args.allowed_chroms else None

    try:
        bed = run_microdna_map(
            r1=args.r1,
            r2=args.r2,
            reference=args.reference,
            output_dir=args.output_dir,
            model_path=args.model_path,
            limit=args.limit,
            threads=args.threads,
            cnvkit_reference=args.cnvkit_reference,
            allowed_chroms=allowed_chroms,
            pipeline_type=args.pipeline,
            fold_change=args.fold_change,
        )
        print(f"[INFO] 检测完成, BED: {bed}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()