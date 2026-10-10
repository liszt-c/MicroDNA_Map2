#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark.py
================

统一入口，按阶段执行 Spike-in 基准测试 (BWA-MEM 统一架构版)。
支持使用逗号分隔指定多个执行阶段，例如:
python benchmark/run_benchmark.py --config benchmark/config_real.yaml --quick --phase 1,2,3,5,7
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BENCHMARK_DIR / "config_real.yaml"


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        print("[ERROR] 配置文件格式错误", file=sys.stderr)
        sys.exit(1)
    return cfg


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, dry_run: bool = False) -> int:
    print(f"[CMD] {' '.join(map(str, cmd))}")
    if dry_run:
        return 0
    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    elapsed = time.time() - start
    print(f"[INFO] 命令{'成功' if result.returncode == 0 else '失败'} (耗时 {elapsed:.2f}s)")
    return result.returncode


def check_dependencies() -> Tuple[bool, List[str]]:
    tools = ["python", "samtools", "bwa"]
    missing = [t for t in tools if shutil.which(t) is None]
    import importlib.util
    for pkg in ["pysam", "yaml", "torch", "matplotlib", "pandas", "seaborn"]:
        if importlib.util.find_spec(pkg) is None:
            missing.append(f"python-{pkg}")
    return len(missing) == 0, missing


def phase0_check(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 0: 环境与参考索引检查 ===")
    ok, missing = check_dependencies()
    if not ok:
        print("[ERROR] 缺少依赖:", ", ".join(missing))
        return 1

    genome = Path(config["genome"]["reference"])
    if not genome.exists():
        print(f"[ERROR] 参考基因组不存在: {genome}")
        return 1
    print(f"[INFO] 统一参考基因组定位: {genome}")

    # 1. 检查 / 创建 samtools faidx (.fai)
    fai_file = Path(str(genome) + ".fai")
    if not fai_file.exists():
        print(f"[INFO] 正在为 {genome.name} 创建 .fai 索引...")
        if not dry_run:
            subprocess.run(["samtools", "faidx", str(genome)], check=True)
    else:
        print(f"[INFO] FASTA .fai 索引已存在: {fai_file.name}")

    # 2. 检查 / 创建 BWA 索引 (Circle-Map 与 MicroDNA Map 全局统一所需)
    bwa_exts = [".amb", ".ann", ".bwt", ".pac", ".sa"]
    if not all(Path(str(genome) + ext).exists() for ext in bwa_exts):
        print(f"[INFO] 正在为 {genome.name} 创建 BWA 索引...")
        if not dry_run:
            subprocess.run(["bwa", "index", str(genome)], check=True)
    else:
        print("[INFO] BWA 索引已存在")

    model = Path(config["detection"]["microdna_map_model"])
    if not model.exists():
        print(f"[WARN] 模型权重文件不存在: {model}")
    else:
        print(f"[INFO] 模型权重定位: {model}")

    print("[INFO] 环境与参考基因组索引检查完毕，全局已就绪\n")
    return 0


def phase1_generate(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 1: 生成模拟 microDNA ===")
    script = BENCHMARK_DIR / "simulate" / "generate_microdna.py"
    if not script.exists():
        return 1
    sim = config["simulation"]
    out_dir = Path(config["experiment"]["output_dir"]) / "truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--genome_path", str(config["genome"]["reference"]),
        "--num_sites", str(sim.get("num_sites", 1000)),
        "--min_len", str(sim.get("min_length", 200)),
        "--max_len", str(sim.get("max_length", 800)),
        "--seed", str(config["experiment"]["seed"]),
        "--output_dir", str(out_dir),
    ]

    if config.get("quick", False):
        bg_chroms = ",".join(config["genome"]["quick_chromosomes"])
    else:
        bg_chroms = ",".join(config["genome"]["background_chroms"])
    cmd += ["--background_chroms", bg_chroms]

    use_real = sim.get("use_real_microdna", False)
    input_dir = sim.get("input_fasta_dir")
    
    if use_real and input_dir:
        input_dir_path = Path(input_dir)
        if input_dir_path.exists():
            cmd += ["--input_fasta_dir", str(input_dir_path)]
            print(f"[INFO] 使用真实 microDNA 序列输入: {input_dir_path}")
        else:
            print(f"[WARN] 真实 microDNA 输入路径不存在: {input_dir_path}，将回退到随机模式。")

    if config["genome"].get("gap_bed"):
        cmd += ["--gap_bed", str(config["genome"]["gap_bed"])]

    return run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)


def phase2_circseq(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 2: 模拟 Circle-seq reads ===")
    script = BENCHMARK_DIR / "simulate" / "simulate_circseq.py"
    if not script.exists():
        return 1
    truth_dir = Path(config["experiment"]["output_dir"]) / "truth"
    out_dir = Path(config["experiment"]["output_dir"]) / "circseq"
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = config["detection"].get("threads", 8)  
    cmd = [
        sys.executable, str(script),
        "--truth_dir", str(truth_dir),
        "--output_dir", str(out_dir),
        "--copy_numbers", ",".join(map(str, config["simulation"]["copy_numbers"])),
        "--read_length", str(config["sequencing"]["read_length"]),
        "--insert_size_mean", str(config["sequencing"]["insert_size_mean"]),
        "--insert_size_std", str(config["sequencing"]["insert_size_std"]),
        "--coverage_per_copy", str(config["circseq"]["coverage_per_copy"]),
        "--junction_enrichment_factor", str(config["circseq"]["junction_enrichment"]),
        "--error_rate", str(config["sequencing"]["error_rate"]),
        "--seed", str(config["experiment"]["seed"]),
        "--threads", str(threads),
    ]
    return run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)


def phase3_wgs(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 3: 模拟 WGS reads ===")
    script = BENCHMARK_DIR / "simulate" / "simulate_wgs.py"
    if not script.exists():
        return 1
    truth_bed = Path(config["experiment"]["output_dir"]) / "truth" / "microdna_truth.bed"
    out_dir = Path(config["experiment"]["output_dir"]) / "wgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = config["detection"].get("threads", 8)
    cmd = [
        sys.executable, str(script),
        "--genome_path", str(config["genome"]["reference"]),
        "--truth_bed", str(truth_bed),
        "--copy_numbers", ",".join(map(str, config["simulation"]["copy_numbers"])),
        "--background_coverage", str(config["simulation"]["background_coverage"]),
        "--read_length", str(config["sequencing"]["read_length"]),
        "--insert_size_mean", str(config["sequencing"]["insert_size_mean"]),
        "--insert_size_std", str(config["sequencing"]["insert_size_std"]),
        "--seed", str(config["experiment"]["seed"]),
        "--output_dir", str(out_dir),
        "--threads", str(threads),
    ]
    if config.get("quick", False):
        cmd += ["--background_chroms", ",".join(config["genome"]["quick_chromosomes"])]
        cmd += ["--max_microdna", str(config["simulation"].get("quick_num_sites", 100))]
    else:
        cmd += ["--background_chroms", ",".join(config["genome"]["background_chroms"])]
    return run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)


def phase4_circle_circseq(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 4: Circle-Map on Circle-seq ===")
    script = BENCHMARK_DIR / "detect" / "run_circlemmap.py"
    if not script.exists():
        return 1
    circseq_dir = Path(config["experiment"]["output_dir"]) / "circseq"
    out_base = Path(config["experiment"]["output_dir"]) / "detect" / "circlemmap"
    threads = config["detection"]["circlemmap_threads"]
    for cn in config["simulation"]["copy_numbers"]:
        r1 = circseq_dir / f"cn{cn}" / "circseq_R1.fastq"
        r2 = circseq_dir / f"cn{cn}" / "circseq_R2.fastq"
        if not (r1.exists() and r2.exists()):
            continue
        out_dir = out_base / f"cn{cn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--r1", str(r1), "--r2", str(r2),
               "--reference", str(config["genome"]["reference"]),
               "--output_dir", str(out_dir), "--threads", str(threads)]
        if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
            return 1
    return 0


def phase5_microdna_wgs(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 5: MicroDNA Map on WGS ===")
    script = BENCHMARK_DIR / "detect" / "run_microdna_map.py"
    if not script.exists():
        return 1
    wgs_dir = Path(config["experiment"]["output_dir"]) / "wgs"
    out_base = Path(config["experiment"]["output_dir"]) / "detect" / "microdna_map"
    threads = config["detection"]["threads"]
    model = config["detection"]["microdna_map_model"]
    limit = config["detection"]["microdna_map_limit"]
    cnvkit_ref = config["genome"].get("cnvkit_reference")
    pipeline_type = config["detection"].get("pipeline", "micro_coverage")
    fold_change = config["detection"].get("fold_change", 1.3)
    
    if config.get("quick", False):
        allowed_chroms = ",".join(config["genome"]["quick_chromosomes"])
    else:
        allowed_chroms = ",".join(config["genome"]["background_chroms"])

    for cn in config["simulation"]["copy_numbers"]:
        r1 = wgs_dir / f"cn{cn}" / f"wgs_{cn}x_R1.fastq"
        r2 = wgs_dir / f"cn{cn}" / f"wgs_{cn}x_R2.fastq"
        if not (r1.exists() and r2.exists()):
            continue
        out_dir = out_base / f"cn{cn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [sys.executable, str(script), "--r1", str(r1), "--r2", str(r2),
               "--reference", str(config["genome"]["reference"]),
               "--output_dir", str(out_dir), "--model_path", str(model),
               "--limit", str(limit), "--threads", str(threads),
               "--allowed_chroms", allowed_chroms,
               "--pipeline", str(pipeline_type),
               "--fold_change", str(fold_change)]
        
        if cnvkit_ref and Path(cnvkit_ref).exists() and pipeline_type == "cnvkit":
            cmd += ["--cnvkit_reference", str(cnvkit_ref)]
            
        if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
            return 1
    return 0


def phase6_circle_wgs(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 6: Circle-Map on WGS (对照) ===")
    script = BENCHMARK_DIR / "detect" / "run_circlemmap.py"
    if not script.exists():
        return 1
    wgs_dir = Path(config["experiment"]["output_dir"]) / "wgs"
    out_base = Path(config["experiment"]["output_dir"]) / "detect" / "circlemmap_wgs"
    threads = config["detection"]["circlemmap_threads"]
    for cn in config["simulation"]["copy_numbers"]:
        r1 = wgs_dir / f"cn{cn}" / f"wgs_{cn}x_R1.fastq"
        r2 = wgs_dir / f"cn{cn}" / f"wgs_{cn}x_R2.fastq"
        if not (r1.exists() and r2.exists()):
            continue
        out_dir = out_base / f"cn{cn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--r1", str(r1), "--r2", str(r2),
               "--reference", str(config["genome"]["reference"]),
               "--output_dir", str(out_dir), "--threads", str(threads)]
        if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
            return 1
    return 0


def phase7_microdna_circseq(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 7: MicroDNA Map on Circle-seq (跨模态评估) ===")
    script = BENCHMARK_DIR / "detect" / "run_microdna_map.py"
    if not script.exists():
        return 1
    circseq_dir = Path(config["experiment"]["output_dir"]) / "circseq"
    out_base = Path(config["experiment"]["output_dir"]) / "detect" / "microdna_circseq"
    threads = config["detection"]["threads"]
    model = config["detection"]["microdna_map_model"]
    limit = config["detection"]["microdna_map_limit"]
    cnvkit_ref = config["genome"].get("cnvkit_reference")
    pipeline_type = config["detection"].get("pipeline", "micro_coverage")
    fold_change = config["detection"].get("fold_change", 1.3)

    for cn in config["simulation"]["copy_numbers"]:
        r1 = circseq_dir / f"cn{cn}" / "circseq_R1.fastq"
        r2 = circseq_dir / f"cn{cn}" / "circseq_R2.fastq"
        if not (r1.exists() and r2.exists()):
            continue
        out_dir = out_base / f"cn{cn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--r1", str(r1), "--r2", str(r2),
               "--reference", str(config["genome"]["reference"]),
               "--output_dir", str(out_dir), "--model_path", str(model),
               "--limit", str(limit), "--threads", str(threads),
               "--pipeline", str(pipeline_type),
               "--fold_change", str(fold_change)]
        if cnvkit_ref and Path(cnvkit_ref).exists() and pipeline_type == "cnvkit":
            cmd += ["--cnvkit_reference", str(cnvkit_ref)]
        if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
            return 1
    return 0


def phase8_evaluate(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 8: 评估 ===")
    truth_bed = Path(config["experiment"]["output_dir"]) / "truth" / "microdna_truth.bed"
    script = BENCHMARK_DIR / "evaluate" / "metrics.py"
    if not script.exists():
        return 1
    eval_base = Path(config["experiment"]["output_dir"]) / "evaluation"
    eval_base.mkdir(parents=True, exist_ok=True)
    lod_threshold = config["evaluation"]["lod_recall_threshold"]
    overlap_threshold = config["evaluation"]["overlap_threshold"]

    tasks = [
        ("circlemmap_circseq", Path(config["experiment"]["output_dir"]) / "detect" / "circlemmap"),
        ("microdna_map_wgs", Path(config["experiment"]["output_dir"]) / "detect" / "microdna_map"),
        ("microdna_map_circseq", Path(config["experiment"]["output_dir"]) / "detect" / "microdna_circseq"),
        ("circlemmap_wgs", Path(config["experiment"]["output_dir"]) / "detect" / "circlemmap_wgs"),
    ]
    for name, detected_dir in tasks:
        out_dir = eval_base / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if detected_dir.exists():
            cmd = [sys.executable, str(script), "--truth_bed", str(truth_bed),
                   "--detected_dir", str(detected_dir), "--output_dir", str(out_dir),
                   "--lod_threshold", str(lod_threshold),
                   "--overlap_threshold", str(overlap_threshold)]
            if run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT) != 0:
                return 1
    return 0


def phase9_visualize(config: dict, dry_run: bool = False) -> int:
    print("\n=== Phase 9: 可视化与报告 ===")
    eval_base = Path(config["experiment"]["output_dir"]) / "evaluation"
    vis_dir = Path(config["experiment"]["output_dir"]) / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    plot_script = BENCHMARK_DIR / "visualize" / "plot_all.py"
    if not plot_script.exists():
        return 1

    lod_cmd = [
        sys.executable, str(plot_script), "--plot", "lod",
        "--microdna_metrics", str(eval_base / "microdna_map_wgs" / "summary_metrics.tsv"),
        "--microdna_circseq_metrics", str(eval_base / "microdna_map_circseq" / "summary_metrics.tsv"),
        "--circseq_metrics", str(eval_base / "circlemmap_circseq" / "summary_metrics.tsv"),
        "--circseq_wgs_metrics", str(eval_base / "circlemmap_wgs" / "summary_metrics.tsv"),
        "--output_dir", str(vis_dir),
        "--lod_threshold", str(config["evaluation"]["lod_recall_threshold"]),
    ]
    comp_cmd = [
        sys.executable, str(plot_script), "--plot", "comparison",
        "--microdna_metrics", str(eval_base / "microdna_map_wgs" / "summary_metrics.tsv"),
        "--microdna_circseq_metrics", str(eval_base / "microdna_map_circseq" / "summary_metrics.tsv"),
        "--circseq_metrics", str(eval_base / "circlemmap_circseq" / "summary_metrics.tsv"),
        "--circseq_wgs_metrics", str(eval_base / "circlemmap_wgs" / "summary_metrics.tsv"),
        "--output_dir", str(vis_dir),
    ]
    run_cmd(lod_cmd, dry_run=dry_run, cwd=PROJECT_ROOT)
    run_cmd(comp_cmd, dry_run=dry_run, cwd=PROJECT_ROOT)

    if not dry_run:
        generate_report(config, eval_base, vis_dir)
    return 0


def generate_report(config: dict, eval_base: Path, vis_dir: Path) -> None:
    report_file = Path(config["experiment"]["output_dir"]) / "final_report.md"
    with open(report_file, "w") as f:
        f.write(f"# {config['experiment']['name']} - Benchmark Report\n\n")
        f.write(f"Seed: {config['experiment']['seed']}\n")
        f.write(f"Copy numbers: {config['simulation']['copy_numbers']}\n\n")
        methods = {
            "MicroDNA Map (WGS)": eval_base / "microdna_map_wgs" / "summary_metrics.tsv",
            "MicroDNA Map (Circle-seq)": eval_base / "microdna_map_circseq" / "summary_metrics.tsv",
            "Circle-Map (Circle-seq)": eval_base / "circlemmap_circseq" / "summary_metrics.tsv",
            "Circle-Map (WGS)": eval_base / "circlemmap_wgs" / "summary_metrics.tsv",
        }
        for method, file in methods.items():
            f.write(f"## {method}\n")
            if file.exists():
                f.write("| Copy Number | Recall | Precision | F1 |\n|---|---|---|---|\n")
                with open(file) as sf:
                    lines = sf.readlines()[1:]
                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) >= 8:
                            f.write(f"| {parts[0]} | {parts[6]} | {parts[7]} | {parts[8]} |\n")
            else:
                f.write("No results.\n")
            f.write("\n")
        f.write("## Visualizations\n")
        f.write(f"See `{vis_dir}`.\n")
    print(f"[INFO] 报告已生成: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一基准测试入口")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase", type=str, default=None,
        help="指定运行的阶段，使用逗号分隔 (例如 '1,2,3,5,7')。如果不提供，默认按顺序执行 0 到 9 的所有阶段。"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 路径自适应绝对化解析：统一锚定到 PROJECT_ROOT，防止由于执行工作目录变动而导致路径断裂
    ref_path = Path(config["genome"]["reference"])
    if not ref_path.is_absolute():
        config["genome"]["reference"] = str((PROJECT_ROOT / ref_path).resolve())

    if config["genome"].get("cnvkit_reference"):
        cnn_path = Path(config["genome"]["cnvkit_reference"])
        if not cnn_path.is_absolute():
            config["genome"]["cnvkit_reference"] = str((PROJECT_ROOT / cnn_path).resolve())

    if config["detection"].get("microdna_map_model"):
        model_path = Path(config["detection"]["microdna_map_model"])
        if not model_path.is_absolute():
            config["detection"]["microdna_map_model"] = str((PROJECT_ROOT / model_path).resolve())

    if args.quick:
        config["quick"] = True
        config["simulation"]["num_sites"] = config["simulation"].get("quick_num_sites", 100)
        config["genome"]["background_chroms"] = config["genome"]["quick_chromosomes"]

    output_root = Path(config["experiment"]["output_dir"])
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
        config["experiment"]["output_dir"] = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    phases = {
        0: phase0_check,
        1: phase1_generate,
        2: phase2_circseq,
        3: phase3_wgs,
        4: phase4_circle_circseq,
        5: phase5_microdna_wgs,
        6: phase6_circle_wgs,
        7: phase7_microdna_circseq,
        8: phase8_evaluate,
        9: phase9_visualize,
    }

    if args.phase is not None:
        try:
            to_run = sorted(list(set(int(p.strip()) for p in args.phase.split(","))))
            for p in to_run:
                if p not in phases:
                    print(f"[ERROR] 无效的 Phase 阶段: {p}。有效阶段为 0 到 9。", file=sys.stderr)
                    sys.exit(1)
        except ValueError:
            print("[ERROR] --phase 参数格式错误，请使用逗号分隔的数字，例如 '1,2,3,5,7'", file=sys.stderr)
            sys.exit(1)
    else:
        to_run = list(range(10))

    total_start = time.time()
    for p in to_run:
        ret = phases[p](config, args.dry_run)
        if ret != 0:
            print(f"[ERROR] Phase {p} 失败")
            sys.exit(1)
        print(f"[INFO] Phase {p} 完成")
    
    if not args.dry_run:
        print(f"[INFO] 全部流程完成，总耗时 {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()